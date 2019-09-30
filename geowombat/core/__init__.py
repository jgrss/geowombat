from .io import to_raster
from .ops import SpatialOperations
from .util import Converters, MapProcesses

extract = SpatialOperations().extract
subset = SpatialOperations().subset
polygons_to_points = Converters().polygons_to_points
moving = MapProcesses().moving

__all__ = ['to_raster',
           'extract',
           'subset',
           'polygons_to_points',
           'moving']
