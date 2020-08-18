from .classifiers import Classifiers
from .transformers import Stackerizer


fit = Classifiers().fit
fit_predict = Classifiers().fit_predict

__all__ = ['fit', 'fit_predict']
