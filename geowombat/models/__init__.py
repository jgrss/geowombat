from .model import GeoWombatClassifier, Predict
from ._crf import time_to_crffeas

predict = Predict().predict

__all__ = ['GeoWombatClassifier', 'predict', 'time_to_crffeas']
