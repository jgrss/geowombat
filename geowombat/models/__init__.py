from .model import CRFClassifier, GeoWombatClassifier, Predict

predict = Predict().predict

__all__ = ['CRFClassifier', 'GeoWombatClassifier', 'predict']
