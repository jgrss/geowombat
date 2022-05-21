from .plotting import Plotting
from .web import GeoDownloads

imshow = Plotting().imshow

__all__ = ['imshow', 'GeoDownloads']
