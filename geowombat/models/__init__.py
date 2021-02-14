from .model import GeoWombatClassifier
from .model import Predict


predict = Predict().predict

__all__ = ['GeoWombatClassifier',
           'predict']
