from .plotting import Plotting
from .web import Downloads

imshow = Plotting().imshow
download = Downloads().download

__all__ = ['imshow', 'download']
