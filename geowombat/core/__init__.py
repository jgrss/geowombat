from .io import apply
from .io import to_raster
from .io import to_vrt
from .io import to_geodataframe
from .sops import SpatialOperations
from .util import Converters
from .util import FileFilters
from .util import MapProcesses
from .vi import VegetationIndices
from .vi import TasseledCap

# Imports intended for package level
extract = SpatialOperations().extract
sample = SpatialOperations().sample
subset = SpatialOperations().subset
clip = SpatialOperations().clip
mask = SpatialOperations().mask
coregister = SpatialOperations().coregister
polygons_to_points = Converters().polygons_to_points
indices_to_coords = Converters().indices_to_coords
coords_to_indices = Converters().coords_to_indices
moving = MapProcesses().moving
norm_diff = VegetationIndices().norm_diff
evi = VegetationIndices().evi
evi2 = VegetationIndices().evi2
nbr = VegetationIndices().nbr
ndvi = VegetationIndices().ndvi
wi = VegetationIndices().wi
tasseled_cap = TasseledCap().tasseled_cap

# Imports intended for module level
parse_filename_dates = FileFilters().parse_filename_dates
parse_wildcard = FileFilters().parse_wildcard

__all__ = ['apply',
           'to_raster',
           'to_vrt',
           'to_geodataframe',
           'extract',
           'sample',
           'subset',
           'clip',
           'mask',
           'coregister',
           'polygons_to_points',
           'indices_to_coords',
           'coords_to_indices',
           'parse_filename_dates',
           'parse_wildcard',
           'moving',
           'norm_diff',
           'evi',
           'evi2',
           'nbr',
           'ndvi',
           'wi',
           'tasseled_cap']
