from .properties import DataProperties
from .plotting import Plotting

imshow = Plotting().imshow

__all__ = ['DataProperties',
           'imshow']
