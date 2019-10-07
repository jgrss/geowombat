from .io import to_raster, parse_wildcard
from .sops import SpatialOperations
from .util import Converters, MapProcesses
from .vi import VegetationIndices

extract = SpatialOperations().extract
subset = SpatialOperations().subset
clip = SpatialOperations().clip
polygons_to_points = Converters().polygons_to_points
moving = MapProcesses().moving
norm_diff = VegetationIndices().norm_diff
evi = VegetationIndices().evi
evi2 = VegetationIndices().evi2
nbr = VegetationIndices().nbr
ndvi = VegetationIndices().ndvi
wi = VegetationIndices().wi

__all__ = ['to_raster',
           'parse_wildcard',
           'extract',
           'subset',
           'clip',
           'polygons_to_points',
           'moving',
           'norm_diff',
           'evi',
           'evi2',
           'nbr',
           'ndvi',
           'wi']
