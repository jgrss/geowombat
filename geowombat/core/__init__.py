from .io import apply, parse_wildcard, to_raster
from .sops import SpatialOperations
from .util import Converters, MapProcesses
from .vi import VegetationIndices
from .vi import TasseledCap

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
tasseled_cap = TasseledCap().tasseled_cap

__all__ = ['apply',
           'parse_wildcard',
           'to_raster',
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
           'wi',
           'tasseled_cap',
           'SurfaceReflectance',
           'gen_pixel_angles']
