from .classifiers import Classifiers

_classifier = Classifiers()

fit = _classifier.fit
fit_predict = _classifier.fit_predict
predict = _classifier.predict

__all__ = ['fit', 'fit_predict', 'predict']

try:
    from .dl_classifiers import (
        TabNetClassifier,
        LTAEClassifier,
        TorchGeoClassifier,
    )
    __all__ += [
        'TabNetClassifier',
        'LTAEClassifier',
        'TorchGeoClassifier',
    ]
except ImportError:
    pass
