from .classifiers import Classifiers

fit = Classifiers().fit
fit_predict = Classifiers().fit_predict
predict = Classifiers().predict

__all__ = ['fit', 'fit_predict','predict']
