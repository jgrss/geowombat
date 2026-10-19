"""Shared PyTorch machinery for per-pixel sequence classifiers.

Provides :class:`TorchSequenceClassifier`, the base for classifiers that learn
from each pixel's ``(time, band)`` series (L-TAE, TempCNN, CNN-BiLSTM).
Subclasses only implement ``_build_module()``.

Requires: ``pip install "geowombat[dl]"``
"""

import copy
import random

import dask.array as da
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from ._base import GeoWombatDLClassifier

LOSSES = ('ce', 'focal')
SCHEDULERS = (None, 'cosine')


def _resolve_device(device):
    """Resolve device string to torch.device."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _set_seed(seed):
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(class_weight, y, n_classes):
    """Per-class weights as a float32 array, or None."""
    if class_weight is None:
        return None
    if isinstance(class_weight, str):
        if class_weight != 'balanced':
            raise ValueError(
                "class_weight must be None, 'balanced' or a "
                "sequence of length n_classes"
            )
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = len(y) / (n_classes * counts)
    else:
        weights = np.asarray(class_weight, dtype=np.float64)
        if weights.shape != (n_classes,):
            raise ValueError(
                f"class_weight must have length {n_classes}; "
                f"got shape {weights.shape}"
            )
    return weights.astype(np.float32)


class _FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weights."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            'alpha', None if alpha is None else torch.as_tensor(alpha)
        )

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


def _make_loss(loss, weights, device):
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}")
    w = None if weights is None else torch.from_numpy(weights).to(device)
    if loss == 'focal':
        return _FocalLoss(alpha=w).to(device)
    return nn.CrossEntropyLoss(weight=w)


def _predict_batches(model, X, batch_size, device):
    """Argmax predictions for a float32 array of series."""
    model.eval()
    out = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            out[i : i + batch_size] = model(xb).argmax(dim=1).cpu().numpy()
    return out


def _train_sequence_model(
    model,
    X,
    y,
    *,
    criterion,
    optimizer,
    max_epochs,
    batch_size,
    device,
    val_fraction=0.0,
    patience=15,
    scheduler=None,
    seed=None,
    grad_clip=1.0,
    verbose=0,
):
    """Train ``model`` on ``(X, y)``; returns the best (1-based) epoch.

    With ``val_fraction > 0`` a random split is held out, the validation
    macro-F1 is tracked, training stops after ``patience`` epochs without
    improvement, and the best weights are restored.
    """
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}")
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = int(round(n * val_fraction)) if val_fraction > 0 else 0
    if n_val < 1 or n - n_val < 1:
        n_val = 0
    val_idx, tr_idx = order[:n_val], order[n_val:]

    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    sched = None
    if scheduler == 'cosine':
        lr0 = optimizer.param_groups[0]['lr']
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr0 * 1e-3
        )

    best_score, best_epoch, best_state, bad = -np.inf, max_epochs, None, 0
    for epoch in range(max_epochs):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item()
            n_batches += 1
        if sched is not None:
            sched.step()

        msg = (
            f"  Epoch {epoch + 1}/{max_epochs} - loss: "
            f"{total / max(n_batches, 1):.4f}"
        )
        if n_val:
            preds = _predict_batches(model, X[val_idx], batch_size, device)
            score = f1_score(y[val_idx], preds, average='macro')
            msg += f" - val macro-F1: {score:.4f}"
            if score > best_score:
                best_score, best_epoch, bad = score, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        if verbose and (n_val or (epoch + 1) % max(1, max_epochs // 10) == 0):
            print(msg)
        if n_val and bad >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


class TorchSequenceClassifier(GeoWombatDLClassifier):
    """Base class for per-pixel sequence classifiers built on PyTorch.

    Handles the whole workflow: input audit and gap filling (see
    :class:`geowombat.ml.GeoWombatDLClassifier`), label rasterisation,
    per-band standardisation, training with optional early stopping, and
    lazy block-wise prediction. Subclasses implement
    ``_build_module(in_channels, n_classes)`` returning an ``nn.Module``
    whose ``forward`` takes ``(batch, time, bands)`` and returns logits.

    Shared parameters (every subclass accepts these as keywords)
    ---------------------------------------------------------------
    max_epochs : int
        Maximum training epochs. Default 50.
    batch_size : int
        Mini-batch size for training and prediction. Default 256.
    lr : float
        Learning rate (AdamW). Default 1e-3.
    weight_decay : float
        AdamW weight decay. Default 0.
    scheduler : {None, 'cosine'}
        Learning-rate schedule. Default None.
    loss : {'ce', 'focal'}
        Cross-entropy or focal loss (gamma=2). Default 'ce'.
    class_weight : None, 'balanced' or sequence
        Per-class loss weights, in the sorted order of the label values.
    val_fraction : float
        Fraction of labelled pixels held out for early stopping. 0
        disables early stopping. Default 0.
    patience : int
        Epochs without validation improvement before stopping.
        Default 15.
    device : str
        'cpu', 'cuda' or 'auto'. Default 'cpu'.
    seed : int or None
        Seed for the split, shuffling and weight initialisation.
    verbose : int
        0 = silent, 1 = progress. Default 0.
    audit : {'warn', 'raise', 'off'}
        Input audit policy. Default 'warn'.
    fill : {'train_median', 'interpolate', 'zero', 'none'}
        Gap-filling strategy. Default 'train_median'.
    """

    def __init__(
        self,
        *,
        max_epochs=50,
        batch_size=256,
        lr=1e-3,
        weight_decay=0.0,
        scheduler=None,
        loss='ce',
        class_weight=None,
        val_fraction=0.0,
        patience=15,
        device='cpu',
        seed=None,
        verbose=0,
        audit='warn',
        fill='train_median',
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.loss = loss
        self.class_weight = class_weight
        self.val_fraction = val_fraction
        self.patience = patience
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.audit = audit
        self.fill = fill
        self._model = None
        self._n_classes = None
        self._n_bands = None
        self._n_timesteps = None
        self._feat_mean = None
        self._feat_std = None
        self.best_epoch_ = None
        self.fitted_ = False

    def _build_module(self, in_channels, n_classes):
        raise NotImplementedError

    def fit(self, data, labels=None, col=None, targ_name="targ", **kwargs):
        self._check_series(data, 'fit')
        filled = self._prepare_fit_data(data)
        X, y = self._extract_training_series(filled, data, labels, col)

        # Standardize per band (across pixels and time)
        self._feat_mean = X.mean(axis=(0, 1))
        self._feat_std = X.std(axis=(0, 1)) + 1e-8
        X = (X - self._feat_mean) / self._feat_std

        _set_seed(self.seed)
        dev = _resolve_device(self.device)
        self._model = self._build_module(self._n_bands, self._n_classes).to(
            dev
        )
        weights = _class_weights(self.class_weight, y, self._n_classes)
        criterion = _make_loss(self.loss, weights, dev)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.best_epoch_ = _train_sequence_model(
            self._model,
            X,
            y,
            criterion=criterion,
            optimizer=optimizer,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device=dev,
            val_fraction=self.val_fraction,
            patience=self.patience,
            scheduler=self.scheduler,
            seed=self.seed,
            verbose=self.verbose,
        )
        self.fitted_ = True
        return self

    def predict(self, data, **kwargs):
        filled, residual = self._prepare_predict_data(data)

        model = self._model
        feat_mean, feat_std = self._feat_mean, self._feat_std
        batch_size, device_str = self.batch_size, self.device

        def _predict_block(block):
            nt, nb, cy, cx = block.shape
            series = block.transpose(2, 3, 0, 1).reshape(cy * cx, nt, nb)
            series = np.nan_to_num(series.astype(np.float32), nan=0.0)
            series = (series - feat_mean) / feat_std
            preds = _predict_batches(
                model, series, batch_size, _resolve_device(device_str)
            )
            return (preds + 1).astype(np.float32).reshape(1, cy, cx)

        result_da = da.map_blocks(
            _predict_block,
            filled.data,
            dtype=np.float32,
            drop_axis=[0, 1],
            new_axis=0,
        )
        return self._finalize_prediction(result_da, filled, residual)
