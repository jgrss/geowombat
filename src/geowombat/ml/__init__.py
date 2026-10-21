from .classifiers import Classifiers

_classifier = Classifiers()

fit = _classifier.fit
fit_predict = _classifier.fit_predict
predict = _classifier.predict

__all__ = ['fit', 'fit_predict', 'predict']

try:
    from ._torch_base import TorchSequenceClassifier
    from .dl_classifiers import (
        LTAEClassifier,
        TabNetClassifier,
        TorchGeoClassifier,
    )
    from .sequence_classifiers import CNNBiLSTMClassifier, TempCNNClassifier

    __all__ += [
        'TorchSequenceClassifier',
        'TabNetClassifier',
        'LTAEClassifier',
        'TorchGeoClassifier',
        'TempCNNClassifier',
        'CNNBiLSTMClassifier',
    ]
except ImportError:
    pass

try:
    from .rocket_classifiers import MiniRocketClassifier

    __all__ += ['MiniRocketClassifier']
except ImportError:
    pass
