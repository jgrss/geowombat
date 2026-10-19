"""Lightweight sequence classifiers for satellite image time series.

TempCNN (Pelletier et al., 2019) and CNN-BiLSTM classify each pixel from
its ``(time, band)`` series. Both build on
:class:`geowombat.ml.TorchSequenceClassifier`, so they share the input
audit, gap filling, training options and block-wise prediction with
:class:`geowombat.ml.LTAEClassifier`.

Requires: ``pip install "geowombat[dl]"``

Example
-------
>>> import geowombat as gw
>>> from geowombat.ml import fit_predict, TempCNNClassifier
>>>
>>> with gw.open(image_list, stack_dim='time', nodata=0) as src:
...     y = fit_predict(src, TempCNNClassifier(seed=0), labels, col='lc')

References
----------
Pelletier, C., Webb, G.I. and Petitjean, F. (2019). Temporal
convolutional neural network for the classification of satellite image
time series. *Remote Sensing*, 11(5), 523.
"""

import torch.nn as nn

from ._torch_base import TorchSequenceClassifier

__all__ = ['TempCNNClassifier', 'CNNBiLSTMClassifier']


class _TempCNNModule(nn.Module):
    """1-D convolutions over time, global average pooling, MLP head.

    Input ``(batch, time, bands)``; output logits ``(batch, n_classes)``.
    """

    def __init__(
        self,
        in_channels,
        n_classes,
        hidden=(64, 128, 256),
        kernel_size=3,
        dropout=0.3,
    ):
        super().__init__()
        layers = []
        prev = in_channels
        for width in hidden:
            layers.extend(
                [
                    nn.Conv1d(
                        prev, width, kernel_size, padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        head_width = max(prev // 2, 8)
        self.head = nn.Sequential(
            nn.Linear(prev, head_width),
            nn.BatchNorm1d(head_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, bands, time)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class _CNNBiLSTMModule(nn.Module):
    """1-D convolution, bidirectional LSTM, last-step linear head.

    Input ``(batch, time, bands)``; output logits ``(batch, n_classes)``.
    """

    def __init__(
        self,
        in_channels,
        n_classes,
        conv_filters=64,
        lstm_hidden=64,
        kernel_size=5,
        dropout=0.3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, conv_filters, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, bands, time)
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)  # (batch, time, filters)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.fc(self.dropout(x))


class TempCNNClassifier(TorchSequenceClassifier):
    """Temporal CNN classifier for satellite image time series.

    Three ``Conv1d -> BatchNorm -> ReLU -> Dropout`` blocks over the time
    axis, global average pooling and a small MLP head. Fast to train and
    a strong baseline for per-pixel crop and land-cover mapping from
    multi-date imagery.

    Parameters
    ----------
    hidden : tuple of int
        Output channels of the convolution blocks. Default (64, 128, 256).
    kernel_size : int
        Temporal kernel size. Default 3.
    dropout : float
        Dropout rate. Default 0.3.
    **kwargs
        Shared training and hygiene options, see
        :class:`geowombat.ml.TorchSequenceClassifier`: ``max_epochs``
        (50), ``batch_size`` (256), ``lr`` (1e-3), ``weight_decay``
        (1e-4), ``scheduler`` ('cosine'), ``loss`` ('focal'),
        ``class_weight``, ``val_fraction`` (0.2), ``patience`` (15),
        ``device``, ``seed``, ``verbose``, ``audit``, ``fill``.

    Example
    -------
    >>> with gw.open(image_list, stack_dim='time', nodata=0) as src:
    ...     y = fit_predict(src, TempCNNClassifier(seed=0), labels, col='lc')

    References
    ----------
    Pelletier, C., Webb, G.I. and Petitjean, F. (2019). Temporal
    convolutional neural network for the classification of satellite
    image time series. *Remote Sensing*, 11(5), 523.
    """

    def __init__(
        self,
        hidden=(64, 128, 256),
        kernel_size=3,
        dropout=0.3,
        *,
        weight_decay=1e-4,
        scheduler='cosine',
        loss='focal',
        val_fraction=0.2,
        **kwargs,
    ):
        super().__init__(
            weight_decay=weight_decay,
            scheduler=scheduler,
            loss=loss,
            val_fraction=val_fraction,
            **kwargs,
        )
        self.hidden = tuple(hidden)
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _build_module(self, in_channels, n_classes):
        return _TempCNNModule(
            in_channels,
            n_classes,
            hidden=self.hidden,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )


class CNNBiLSTMClassifier(TorchSequenceClassifier):
    """CNN + bidirectional LSTM classifier for satellite image time series.

    A temporal convolution extracts local features, a bidirectional LSTM
    reads the sequence in both directions, and the last step feeds a
    linear head.

    Parameters
    ----------
    conv_filters : int
        Output channels of the convolution. Default 64.
    lstm_hidden : int
        Hidden size per LSTM direction. Default 64.
    kernel_size : int
        Temporal kernel size. Default 5.
    dropout : float
        Dropout before the head. Default 0.3.
    **kwargs
        Shared training and hygiene options, see
        :class:`geowombat.ml.TorchSequenceClassifier`: ``max_epochs``
        (50), ``batch_size`` (256), ``lr`` (1e-3), ``weight_decay``
        (1e-4), ``scheduler`` ('cosine'), ``loss`` ('focal'),
        ``class_weight``, ``val_fraction`` (0.2), ``patience`` (15),
        ``device``, ``seed``, ``verbose``, ``audit``, ``fill``.

    Example
    -------
    >>> with gw.open(image_list, stack_dim='time', nodata=0) as src:
    ...     y = fit_predict(
    ...         src, CNNBiLSTMClassifier(seed=0), labels, col='lc'
    ...     )
    """

    def __init__(
        self,
        conv_filters=64,
        lstm_hidden=64,
        kernel_size=5,
        dropout=0.3,
        *,
        weight_decay=1e-4,
        scheduler='cosine',
        loss='focal',
        val_fraction=0.2,
        **kwargs,
    ):
        super().__init__(
            weight_decay=weight_decay,
            scheduler=scheduler,
            loss=loss,
            val_fraction=val_fraction,
            **kwargs,
        )
        self.conv_filters = conv_filters
        self.lstm_hidden = lstm_hidden
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _build_module(self, in_channels, n_classes):
        return _CNNBiLSTMModule(
            in_channels,
            n_classes,
            conv_filters=self.conv_filters,
            lstm_hidden=self.lstm_hidden,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
