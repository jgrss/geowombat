from .model import CloudClassifier, GeoWombatClassifier, Predict

predict = Predict().predict

__all__ = ['CloudClassifier', 'GeoWombatClassifier', 'predict']
