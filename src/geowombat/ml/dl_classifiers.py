"""Deep learning classifiers for geowombat.

Provides PyTorch-based classifiers that integrate with the existing
``geowombat.ml.fit()``, ``predict()``, and ``fit_predict()`` API.

Requires: ``pip install geowombat[dl]``

Example
-------
>>> import geowombat as gw
>>> from geowombat.data import l8_224078_20200518
>>> from geowombat.data import l8_224078_20200518_polygons
>>> from geowombat.ml import fit_predict
>>> from geowombat.ml.dl_classifiers import TabNetClassifier
>>>
>>> with gw.open(l8_224078_20200518) as src:
...     y = fit_predict(src, TabNetClassifier(),
...         l8_224078_20200518_polygons, col='lc')
"""

import warnings
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import xarray as xr


def _get_nodata_mask(data):
    """Return a 2-D boolean mask (y, x) that is True for nodata pixels.

    Prefers the ``_nodata_mask`` coordinate (warped from the original
    file) when available.  Falls back to value comparison using
    ``nodatavals`` from attributes.
    """
    if '_nodata_mask' in data.coords:
        return data.coords['_nodata_mask']

    nodatavals = data.attrs.get("nodatavals")
    if not nodatavals or len(nodatavals) == 0:
        return None

    src_nd = nodatavals[0]
    if isinstance(src_nd, float) and np.isnan(src_nd):
        _is_nd = lambda a: a.isnull()
    else:
        _is_nd = lambda a: a == src_nd

    if data.ndim == 4:
        return _is_nd(data.isel(time=0)).any(dim='band')
    else:
        return _is_nd(data).any(dim='band')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError(
        "Deep learning classifiers require PyTorch. "
        "Install with: pip install torch "
        "or: pip install geowombat[dl]"
    ) from e

from .. import polygon_to_array


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _resolve_device(device):
    """Resolve device string to torch.device."""
    if device == 'auto':
        return torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    return torch.device(device)


def _rasterize_labels(data, labels, col):
    """Rasterize vector labels onto the data grid.

    Parameters
    ----------
    data : xarray.DataArray
        Reference raster (band, y, x) or (time, band, y, x).
    labels : str, Path, or GeoDataFrame
        Vector labels.
    col : str
        Column name with class values.

    Returns
    -------
    label_array : numpy.ndarray
        2D array (y, x) with integer class labels (0 = nodata).
    n_classes : int
        Number of unique non-zero classes.
    """
    if isinstance(labels, (str, Path)):
        labels = gpd.read_file(labels)

    labels = labels.copy()

    # Encode labels to 1-based integers (0 = nodata)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(labels[col])
    labels[col] = le.transform(labels[col]) + 1

    # Ensure data is dask-backed (polygon_to_array needs .chunksize)
    if not hasattr(data.data, 'chunksize'):
        data = data.chunk(
            {'y': min(256, data.sizes['y']),
             'x': min(256, data.sizes['x'])}
        )

    label_raster = polygon_to_array(labels, col=col, data=data)
    label_arr = label_raster.values.squeeze()

    n_classes = len(le.classes_)

    return label_arr, n_classes


def _train_loop(model, dataloader, criterion, optimizer, epochs,
                device, verbose=0):
    """Standard PyTorch training loop."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg:.4f}")


def _extract_patches(data_np, label_arr, patch_size, max_patches=500,
                     min_labeled_frac=0.001):
    """Extract training patches from labeled regions.

    Parameters
    ----------
    data_np : numpy.ndarray
        Image data (bands, y, x).
    label_arr : numpy.ndarray
        Label raster (y, x) with 0 = nodata.
    patch_size : int
        Height/width of square patches.
    max_patches : int
        Maximum number of patches to extract.
    min_labeled_frac : float
        Minimum fraction of labeled pixels in a patch.

    Returns
    -------
    patches : numpy.ndarray
        (N, bands, patch_size, patch_size)
    patch_labels : numpy.ndarray
        (N, patch_size, patch_size) with 0-based class labels.
    """
    n_bands, h, w = data_np.shape
    half = patch_size // 2

    # Find all labeled pixel locations
    labeled_ys, labeled_xs = np.where(label_arr > 0)
    if len(labeled_ys) == 0:
        raise ValueError("No labeled pixels found in the data extent.")

    # Sample patch centers from labeled pixels
    n_candidates = min(max_patches * 3, len(labeled_ys))
    idx = np.random.choice(len(labeled_ys), n_candidates, replace=True)

    patches = []
    patch_labels = []
    for i in idx:
        cy, cx = labeled_ys[i], labeled_xs[i]
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = y0 + patch_size
        x1 = x0 + patch_size

        if y1 > h:
            y0, y1 = h - patch_size, h
        if x1 > w:
            x0, x1 = w - patch_size, w
        if y0 < 0 or x0 < 0:
            continue

        lbl_patch = label_arr[y0:y1, x0:x1]
        frac = np.mean(lbl_patch > 0)
        if frac < min_labeled_frac:
            continue

        patches.append(data_np[:, y0:y1, x0:x1])
        # Convert to 0-based (subtract 1), keep nodata as 255
        lbl_zero = lbl_patch.copy().astype(np.int64)
        lbl_zero[lbl_patch > 0] -= 1
        lbl_zero[lbl_patch == 0] = 255  # ignore index
        patch_labels.append(lbl_zero)

        if len(patches) >= max_patches:
            break

    if len(patches) == 0:
        raise ValueError(
            f"No valid patches of size {patch_size} found. "
            "Try a smaller patch_size or check label coverage."
        )

    return np.stack(patches), np.stack(patch_labels)


def _sliding_window_predict(model, data_np, n_classes, patch_size,
                            stride, device):
    """Sliding window inference over a full image.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    data_np : numpy.ndarray
        (bands, H, W) input image.
    n_classes : int
        Number of output classes.
    patch_size : int
        Patch size for inference.
    stride : int
        Stride between patches.
    device : torch.device
        Compute device.

    Returns
    -------
    predictions : numpy.ndarray
        (H, W) with 1-based class labels.
    """
    model.eval()
    n_bands, h, w = data_np.shape

    # Accumulate logits and counts for averaging overlaps
    logit_sum = np.zeros((n_classes, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + patch_size, h)
                x1 = min(x0 + patch_size, w)
                # Adjust start if patch extends beyond image
                y0_adj = max(0, y1 - patch_size)
                x0_adj = max(0, x1 - patch_size)

                patch = data_np[:, y0_adj:y1, x0_adj:x1]
                ph, pw = patch.shape[1], patch.shape[2]

                # Pad if needed
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros(
                        (n_bands, patch_size, patch_size),
                        dtype=np.float32,
                    )
                    padded[:, :ph, :pw] = patch
                    patch = padded

                t = torch.from_numpy(
                    patch[np.newaxis]
                ).float().to(device)

                out = model(t)
                # Handle dict output (some torchgeo models)
                if isinstance(out, dict):
                    out = out.get('out', list(out.values())[0])
                logits = out.cpu().numpy()[0]  # (n_classes, ps, ps)

                # Place back (only the valid region)
                logit_sum[
                    :, y0_adj:y1, x0_adj:x1
                ] += logits[:, :ph, :pw]
                count[y0_adj:y1, x0_adj:x1] += 1.0

    # Average and argmax
    count = np.maximum(count, 1.0)
    avg_logits = logit_sum / count[np.newaxis, :, :]
    preds = np.argmax(avg_logits, axis=0)

    # Convert to 1-based
    return (preds + 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GeoWombatDLClassifier:
    """Base class for deep learning classifiers in geowombat.

    Subclasses must implement ``fit()`` and ``predict()``.
    The ``_is_gw_dl_classifier`` attribute is used by
    ``Classifiers.fit()`` / ``predict()`` to delegate to these
    methods instead of the sklearn_xarray pipeline.
    """
    _is_gw_dl_classifier = True

    def fit(self, data, labels=None, col=None, targ_name="targ",
            **kwargs):
        raise NotImplementedError

    def predict(self, data, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TabNet
# ---------------------------------------------------------------------------

class TabNetClassifier(GeoWombatDLClassifier):
    """TabNet attention-based classifier for pixel-wise classification.

    Wraps pytorch-tabnet internally. Works on tabular (samples, features)
    data — each pixel is one sample with spectral bands as features.

    Parameters
    ----------
    max_epochs : int
        Training epochs. Default 50.
    batch_size : int
        Mini-batch size. Default 1024.
    patience : int
        Early stopping patience. Default 10.
    device : str
        'cpu', 'cuda', or 'auto'. Default 'cpu'.
    verbose : int
        0 = silent, 1 = progress. Default 0.
    tabnet_params : dict
        Extra kwargs passed to pytorch_tabnet.TabNetClassifier.

    Example
    -------
    >>> with gw.open(l8_224078_20200518) as src:
    ...     y = fit_predict(src, TabNetClassifier(), labels, col='lc')
    """

    def __init__(self, max_epochs=50, batch_size=1024, patience=10,
                 device='cpu', verbose=0, **tabnet_params):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.verbose = verbose
        self.tabnet_params = tabnet_params
        self._model = None
        self._n_classes = None
        self._feat_mean = None
        self._feat_std = None
        self.fitted_ = False

    def fit(self, data, labels=None, col=None, targ_name="targ",
            **kwargs):
        try:
            from pytorch_tabnet.tab_model import \
                TabNetClassifier as _TabNet
        except ImportError as e:
            raise ImportError(
                "TabNetClassifier requires pytorch-tabnet. "
                "Install with: pip install pytorch-tabnet"
            ) from e

        # Get image data as numpy (band, y, x)
        data_np = data.values
        if data_np.ndim == 4:
            # (time, band, y, x) -> flatten to (time*band, y, x)
            nt, nb, ny, nx = data_np.shape
            data_np = data_np.reshape(nt * nb, ny, nx)

        n_features = data_np.shape[0]

        # Rasterize labels
        label_arr, self._n_classes = _rasterize_labels(
            data, labels, col
        )

        # Stack to (samples, features)
        # data_np: (bands, y, x) -> (y*x, bands)
        pixels = data_np.reshape(n_features, -1).T  # (n_pixels, n_feat)
        flat_labels = label_arr.ravel()  # (n_pixels,)

        # Filter to labeled pixels (label > 0)
        mask = flat_labels > 0
        X_train = pixels[mask].astype(np.float32)
        y_train = (flat_labels[mask] - 1).astype(np.int64)  # 0-based

        # Standardize features (important for raw DN values)
        self._feat_mean = X_train.mean(axis=0)
        self._feat_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - self._feat_mean) / self._feat_std

        device_name = self.device
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._model = _TabNet(
            device_name=device_name,
            verbose=self.verbose,
            **self.tabnet_params,
        )

        fit_kwargs = {
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'batch_size': min(self.batch_size, len(X_train)),
        }
        # For small datasets, use training data as eval set
        # to enable early stopping without a split
        if len(X_train) < 200:
            fit_kwargs['eval_set'] = [(X_train, y_train)]
            fit_kwargs['eval_metric'] = ['accuracy']

        self._model.fit(X_train, y_train, **fit_kwargs)
        self.fitted_ = True

    def predict(self, data, **kwargs):
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        # Flatten time*band if multi-temporal
        if data.ndim == 4:
            slices = [
                data.isel(time=i).drop_vars('time')
                for i in range(data.sizes['time'])
            ]
            data_flat = xr.concat(slices, dim='band')
        else:
            data_flat = data

        # Ensure dask-backed with band as single chunk
        if not hasattr(data_flat.data, 'dask'):
            data_flat = data_flat.chunk({
                'band': -1,
                'y': min(512, data_flat.sizes['y']),
                'x': min(512, data_flat.sizes['x']),
            })
        else:
            data_flat = data_flat.chunk({'band': -1})

        feat_mean = self._feat_mean
        feat_std = self._feat_std
        tabnet_model = self._model

        def _predict_block(block):
            n_feat, cy, cx = block.shape
            pixels = block.reshape(n_feat, -1).T.astype(np.float32)
            pixels = (pixels - feat_mean) / feat_std
            preds = tabnet_model.predict(pixels)
            return (preds + 1).astype(np.float32).reshape(1, cy, cx)

        result_da = da.map_blocks(
            _predict_block,
            data_flat.data,
            dtype=np.float32,
            drop_axis=0,
            new_axis=0,
        )

        result = xr.DataArray(
            result_da,
            dims=("band", "y", "x"),
            coords={
                "band": ["targ"],
                "y": data.y,
                "x": data.x,
            },
        )

        # Mask nodata pixels in prediction output
        nd_mask = _get_nodata_mask(data)
        if nd_mask is not None:
            result = xr.where(nd_mask, np.nan, result)

        result = result.assign_attrs(**data.attrs)
        return result


# ---------------------------------------------------------------------------
# L-TAE (Lightweight Temporal Attention Encoder)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(
                position * div_term[:d_model // 2]
            )
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, n_timesteps):
        return self.pe[:, :n_timesteps, :]


class _LTAEModule(nn.Module):
    """Lightweight Temporal Attention Encoder.

    Based on Garnot & Landrieu (2020). Takes per-pixel temporal
    sequences and produces a single feature vector via multi-head
    temporal attention with a learned master query.

    Parameters
    ----------
    in_channels : int
        Input features per timestep (spectral bands).
    n_classes : int
        Number of output classes.
    n_head : int
        Number of attention heads.
    d_k : int
        Key dimension per head.
    d_model : int
        Internal embedding dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, in_channels, n_classes, n_head=4, d_k=32,
                 d_model=128, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_model = d_model

        # Input embedding
        self.inconv = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_enc = _PositionalEncoding(d_model)

        # Learned master query per head
        self.query = nn.Parameter(torch.randn(n_head, d_k))

        # Key projection (shared across heads, output split)
        self.key_proj = nn.Linear(d_model, n_head * d_k)

        # Value projection
        self.value_proj = nn.Linear(d_model, n_head * d_k)

        # Output MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(n_head * d_k, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self.scale = d_k ** 0.5

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            (batch, time, in_channels)

        Returns
        -------
        logits : Tensor
            (batch, n_classes)
        """
        batch, T, _ = x.shape

        # Embed input
        x = self.inconv(x)  # (B, T, d_model)

        # Add positional encoding
        x = x + self.pos_enc(T)

        # Keys and Values: (B, T, n_head, d_k)
        K = self.key_proj(x).view(batch, T, self.n_head, self.d_k)
        V = self.value_proj(x).view(batch, T, self.n_head, self.d_k)

        # Transpose for attention: (B, n_head, T, d_k)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Master query: (1, n_head, 1, d_k)
        Q = self.query.unsqueeze(0).unsqueeze(2)

        # Attention scores: (B, n_head, 1, T)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum: (B, n_head, 1, d_k)
        out = torch.matmul(attn, V)

        # Flatten heads: (B, n_head * d_k)
        out = out.squeeze(2).reshape(batch, self.n_head * self.d_k)

        return self.out_mlp(out)


class LTAEClassifier(GeoWombatDLClassifier):
    """L-TAE classifier for satellite image time series.

    Expects multi-temporal data. Uses temporal attention to
    classify each pixel based on its spectral trajectory.

    Parameters
    ----------
    n_head : int
        Number of attention heads. Default 4.
    d_k : int
        Key dimension per head. Default 32.
    d_model : int
        Internal embedding dimension. Default 128.
    dropout : float
        Dropout rate. Default 0.1.
    max_epochs : int
        Training epochs. Default 50.
    batch_size : int
        Mini-batch size. Default 256.
    lr : float
        Learning rate. Default 1e-3.
    device : str
        'cpu', 'cuda', or 'auto'. Default 'cpu'.
    verbose : int
        0 = silent, 1 = progress. Default 0.

    Example
    -------
    >>> with gw.open([img1, img2, img3], stack_dim='time') as src:
    ...     y = fit_predict(src, LTAEClassifier(),
    ...         labels, col='lc', temporal_mode='flatten')
    """

    def __init__(self, n_head=4, d_k=32, d_model=128, dropout=0.1,
                 max_epochs=50, batch_size=256, lr=1e-3,
                 device='cpu', verbose=0):
        self.n_head = n_head
        self.d_k = d_k
        self.d_model = d_model
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self._model = None
        self._n_classes = None
        self._n_bands = None
        self._n_timesteps = None
        self._feat_mean = None
        self._feat_std = None
        self.fitted_ = False

    def fit(self, data, labels=None, col=None, targ_name="targ",
            **kwargs):
        if 'time' not in data.dims:
            raise ValueError(
                "LTAEClassifier requires multi-temporal data with a "
                "'time' dimension. Open data with stack_dim='time'."
            )

        # Infer n_bands and n_timesteps from data dimensions
        self._n_timesteps = data.sizes['time']
        self._n_bands = data.sizes['band']

        # Get data as (time, band, y, x)
        data_np = data.values  # (time, band, y, x)
        nt, nb, ny, nx = data_np.shape

        # Rasterize labels
        label_arr, self._n_classes = _rasterize_labels(
            data, labels, col
        )

        # Reshape to (pixels, time, bands)
        # data_np: (time, band, y, x) -> (y*x, time, band)
        pixels = data_np.transpose(2, 3, 0, 1).reshape(
            ny * nx, nt, nb
        )
        flat_labels = label_arr.ravel()

        # Filter to labeled pixels
        mask = flat_labels > 0
        X_train = pixels[mask].astype(np.float32)
        y_train = (flat_labels[mask] - 1).astype(np.int64)

        # Standardize features per band (across time and pixels)
        self._feat_mean = X_train.mean(axis=(0, 1))
        self._feat_std = X_train.std(axis=(0, 1)) + 1e-8
        X_train = (
            (X_train - self._feat_mean) / self._feat_std
        )

        dev = _resolve_device(self.device)

        self._model = _LTAEModule(
            in_channels=nb,
            n_classes=self._n_classes,
            n_head=self.n_head,
            d_k=self.d_k,
            d_model=self.d_model,
            dropout=self.dropout,
        ).to(dev)

        dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr
        )

        _train_loop(
            self._model, loader, criterion, optimizer,
            self.max_epochs, dev, self.verbose,
        )
        self.fitted_ = True

    def predict(self, data, **kwargs):
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        if data.ndim != 4:
            raise ValueError(
                "LTAEClassifier.predict() requires time-dimensioned "
                "data (time, band, y, x)."
            )

        # Ensure dask-backed with time and band as single chunks
        if not hasattr(data.data, 'dask'):
            data = data.chunk({
                'time': -1, 'band': -1,
                'y': min(512, data.sizes['y']),
                'x': min(512, data.sizes['x']),
            })
        else:
            data = data.chunk({'time': -1, 'band': -1})

        feat_mean = self._feat_mean
        feat_std = self._feat_std
        ltae_model = self._model
        batch_size = self.batch_size
        device_str = self.device

        def _predict_block(block):
            # block: (time, band, y, x) numpy array
            nt, nb, cy, cx = block.shape
            dev = _resolve_device(device_str)

            # Reshape to (pixels, time, bands)
            pixels = block.transpose(
                2, 3, 0, 1
            ).reshape(cy * cx, nt, nb).astype(np.float32)
            pixels = (pixels - feat_mean) / feat_std

            ltae_model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(pixels), batch_size):
                    batch = torch.from_numpy(
                        pixels[i:i + batch_size]
                    ).to(dev)
                    logits = ltae_model(batch)
                    preds.append(
                        logits.argmax(dim=1).cpu().numpy()
                    )

            pred_flat = np.concatenate(preds)
            pred_flat = (pred_flat + 1).astype(np.float32)
            return pred_flat.reshape(1, cy, cx)

        result_da = da.map_blocks(
            _predict_block,
            data.data,
            dtype=np.float32,
            drop_axis=[0, 1],
            new_axis=0,
        )

        result = xr.DataArray(
            result_da,
            dims=("band", "y", "x"),
            coords={
                "band": ["targ"],
                "y": data.y,
                "x": data.x,
            },
        )

        # Mask nodata pixels in prediction output
        nd_mask = _get_nodata_mask(data)
        if nd_mask is not None:
            result = xr.where(nd_mask, np.nan, result)

        result = result.assign_attrs(**data.attrs)
        return result


# ---------------------------------------------------------------------------
# TorchGeo segmentation wrapper
# ---------------------------------------------------------------------------

class TorchGeoClassifier(GeoWombatDLClassifier):
    """Wrapper for TorchGeo segmentation models (U-Net, DeepLabV3+, etc.).

    Uses pre-trained encoder weights from TorchGeo for satellite-
    specific feature extraction. Patch-based training and sliding
    window inference.

    Parameters
    ----------
    model : str
        Segmentation model name: 'unet', 'deeplabv3+', 'fcn'.
        Default 'unet'.
    backbone : str
        Encoder backbone (timm model name). Default 'resnet18'.
    weights : str or None
        TorchGeo weight name, e.g.
        'ResNet18_Weights.SENTINEL2_RGB_MOCO'. None for random init.
    patch_size : int
        Training/inference patch size. Default 64.
    stride : int or None
        Inference stride. Default patch_size // 2.
    max_patches : int
        Max training patches to extract. Default 500.
    bands : list or None
        Band indices or names to select from data.
        If None, uses all bands and validates count.
    max_epochs : int
        Training epochs. Default 50.
    batch_size : int
        Mini-batch size. Default 8.
    lr : float
        Learning rate. Default 1e-3.
    device : str
        'cpu', 'cuda', or 'auto'. Default 'cpu'.
    verbose : int
        0 = silent, 1 = progress. Default 0.

    Example
    -------
    >>> clf = TorchGeoClassifier(
    ...     model='unet', backbone='resnet18',
    ...     weights='ResNet18_Weights.SENTINEL2_RGB_MOCO',
    ...     patch_size=64, bands=[1, 2, 3],
    ... )
    >>> with gw.open(image) as src:
    ...     y = fit_predict(src, clf, labels, col='lc')
    """

    def __init__(self, model='unet', backbone='resnet18', weights=None,
                 patch_size=64, stride=None, max_patches=500,
                 bands=None, max_epochs=50, batch_size=8, lr=1e-3,
                 device='cpu', verbose=0):
        self.model_name = model
        self.backbone = backbone
        self.weights = weights
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2
        self.max_patches = max_patches
        self.bands = bands
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self._model = None
        self._n_classes = None
        self.fitted_ = False

    @property
    def expected_bands(self):
        """Return expected input band info from model weights."""
        if self.weights is None:
            return {'in_chans': None, 'info': 'No weights specified'}
        try:
            from torchgeo.models import get_model_weights
            w = get_model_weights(self.backbone)
            for member in w:
                if member.name == self.weights.split('.')[-1]:
                    meta = member.meta
                    return {
                        'in_chans': meta.get('in_chans'),
                        'meta': meta,
                    }
            return {'in_chans': None, 'info': 'Weight not found'}
        except Exception as exc:
            return {'in_chans': None, 'error': str(exc)}

    def _select_bands(self, data_np, data):
        """Select and reorder bands based on self.bands."""
        if self.bands is None:
            return data_np

        if isinstance(self.bands[0], (int, np.integer)):
            indices = [b - 1 if b > 0 else b for b in self.bands]
            return data_np[indices]
        else:
            band_names = list(data.band.values)
            indices = [band_names.index(b) for b in self.bands]
            return data_np[indices]

    def _build_model(self, in_channels, n_classes, device):
        """Build the segmentation model."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "TorchGeoClassifier requires "
                "segmentation-models-pytorch. "
                "Install with: pip install "
                "segmentation-models-pytorch"
            ) from e

        # Load encoder weights from torchgeo if specified
        encoder_weights = None
        if self.weights:
            try:
                from torchgeo.models import get_model_weights
                w_enum = get_model_weights(self.backbone)
                weight_name = self.weights.split('.')[-1]
                for member in w_enum:
                    if member.name == weight_name:
                        encoder_weights = member
                        break
            except Exception:
                warnings.warn(
                    f"Could not load TorchGeo weights "
                    f"'{self.weights}'. Using random init.",
                    UserWarning,
                )

        model_map = {
            'unet': smp.Unet,
            'deeplabv3+': smp.DeepLabV3Plus,
            'deeplabv3': smp.DeepLabV3,
            'fcn': smp.FPN,
        }

        model_cls = model_map.get(self.model_name.lower())
        if model_cls is None:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Choose from: {list(model_map.keys())}"
            )

        # Build model - use timm encoder
        model_kwargs = {
            'encoder_name': self.backbone,
            'in_channels': in_channels,
            'classes': n_classes,
        }

        # If we have torchgeo weights, we need to handle
        # encoder weight loading separately
        if encoder_weights is not None:
            # Build without pretrained encoder, load weights after
            model_kwargs['encoder_weights'] = None
            model = model_cls(**model_kwargs)

            # Load the torchgeo pretrained encoder weights
            try:
                state_dict = encoder_weights.get_state_dict(
                    progress=True
                )
                # Filter to only encoder keys that match
                encoder_sd = {}
                model_sd = model.encoder.state_dict()
                for k, v in state_dict.items():
                    # Remove prefix if present
                    clean_k = k.replace('backbone.', '').replace(
                        'model.', ''
                    )
                    if clean_k in model_sd:
                        if v.shape == model_sd[clean_k].shape:
                            encoder_sd[clean_k] = v
                if encoder_sd:
                    model.encoder.load_state_dict(
                        encoder_sd, strict=False
                    )
                    if self.verbose:
                        print(
                            f"  Loaded {len(encoder_sd)} encoder "
                            f"weight tensors from TorchGeo"
                        )
            except Exception as exc:
                warnings.warn(
                    f"Failed to load encoder weights: {exc}",
                    UserWarning,
                )
        else:
            model = model_cls(**model_kwargs)

        return model.to(device)

    def fit(self, data, labels=None, col=None, targ_name="targ",
            **kwargs):
        # Get data as (bands, y, x) — use first time step if temporal
        data_np = data.values
        if data_np.ndim == 4:
            data_np = data_np[0]  # first time step
        data_np = data_np.astype(np.float32)

        # Select bands
        data_np = self._select_bands(data_np, data)
        in_channels = data_np.shape[0]

        # Validate band count against weights
        if self.weights and self.bands is None:
            exp = self.expected_bands.get('in_chans')
            if exp is not None and exp != in_channels:
                raise ValueError(
                    f"Model expects {exp} input channels but data "
                    f"has {in_channels} bands. Use the `bands` "
                    f"parameter to select/reorder bands, or use "
                    f"weights trained on {in_channels} bands."
                )

        # Rasterize labels
        label_arr, self._n_classes = _rasterize_labels(
            data, labels, col
        )

        # Extract training patches
        patches, patch_labels = _extract_patches(
            data_np, label_arr, self.patch_size, self.max_patches,
        )

        dev = _resolve_device(self.device)

        # Build model
        self._model = self._build_model(
            in_channels, self._n_classes, dev
        )

        # Create dataloader
        dataset = TensorDataset(
            torch.from_numpy(patches).float(),
            torch.from_numpy(patch_labels).long(),
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
        )

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr
        )

        _train_loop(
            self._model, loader, criterion, optimizer,
            self.max_epochs, dev, self.verbose,
        )
        self.fitted_ = True

    def predict(self, data, **kwargs):
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        # Select bands and flatten time at the DataArray level
        if data.ndim == 4:
            data_3d = data.isel(time=0).drop_vars('time')
        else:
            data_3d = data

        if self.bands is not None:
            if isinstance(self.bands[0], (int, np.integer)):
                indices = [b - 1 if b > 0 else b for b in self.bands]
                data_3d = data_3d.isel(band=indices)
            else:
                data_3d = data_3d.sel(band=self.bands)

        # Ensure dask-backed with band as single chunk
        if not hasattr(data_3d.data, 'dask'):
            data_3d = data_3d.chunk({
                'band': -1,
                'y': min(512, data_3d.sizes['y']),
                'x': min(512, data_3d.sizes['x']),
            })
        else:
            data_3d = data_3d.chunk({'band': -1})

        seg_model = self._model
        n_classes = self._n_classes
        patch_size = self.patch_size
        stride = self.stride
        device_str = self.device

        def _predict_block(block):
            # block: (band, y, x) numpy array
            block_np = block.astype(np.float32)
            dev = _resolve_device(device_str)
            pred_2d = _sliding_window_predict(
                seg_model, block_np, n_classes,
                patch_size, stride, dev,
            )
            return pred_2d[np.newaxis].astype(np.float32)

        result_da = da.map_blocks(
            _predict_block,
            data_3d.data,
            dtype=np.float32,
            drop_axis=0,
            new_axis=0,
        )

        result = xr.DataArray(
            result_da,
            dims=("band", "y", "x"),
            coords={
                "band": ["targ"],
                "y": data.y,
                "x": data.x,
            },
        )

        # Mask nodata pixels in prediction output
        nd_mask = _get_nodata_mask(data)
        if nd_mask is not None:
            result = xr.where(nd_mask, np.nan, result)

        result = result.assign_attrs(**data.attrs)
        return result
