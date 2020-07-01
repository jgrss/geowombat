from ..backends import to_crs
from ..backends import transform_crs
from .io import apply
from .io import to_raster
from .io import to_vrt
from .sops import SpatialOperations
from .conversion import Converters
from .util import MapProcesses
from .vi import VegetationIndices
from .vi import TasseledCap

# Imports intended for module level
from .util import sort_images_by_date

# Imports intended for package level
extract = SpatialOperations().extract
sample = SpatialOperations().sample
calc_area = SpatialOperations().calc_area
subset = SpatialOperations().subset
clip = SpatialOperations().clip
mask = SpatialOperations().mask
coregister = SpatialOperations().coregister
lonlat_to_xy = Converters().lonlat_to_xy
xy_to_lonlat = Converters().xy_to_lonlat
polygons_to_points = Converters().polygons_to_points
indices_to_coords = Converters().indices_to_coords
coords_to_indices = Converters().coords_to_indices
dask_to_xarray = Converters().dask_to_xarray
ndarray_to_xarray = Converters().ndarray_to_xarray
array_to_polygon = Converters().array_to_polygon
polygon_to_array = Converters().polygon_to_array
geodataframe_to_array = Converters().geodataframe_to_array
to_geodataframe = Converters().to_geodataframe
moving = MapProcesses().moving
norm_diff = VegetationIndices().norm_diff
avi = VegetationIndices().avi
evi = VegetationIndices().evi
evi2 = VegetationIndices().evi2
nbr = VegetationIndices().nbr
ndvi = VegetationIndices().ndvi
wi = VegetationIndices().wi
tasseled_cap = TasseledCap().tasseled_cap

__all__ = ['apply',
           'to_crs',
           'transform_crs',
           'to_raster',
           'to_vrt',
           'to_geodataframe',
           'geodataframe_to_array',
           'extract',
           'sample',
           'calc_area',
           'subset',
           'clip',
           'mask',
           'coregister',
           'lonlat_to_xy',
           'xy_to_lonlat',
           'polygons_to_points',
           'indices_to_coords',
           'coords_to_indices',
           'dask_to_xarray',
           'ndarray_to_xarray',
           'array_to_polygon',
           'polygon_to_array',
           'sort_images_by_date',
           'moving',
           'norm_diff',
           'avi',
           'evi',
           'evi2',
           'nbr',
           'ndvi',
           'wi',
           'tasseled_cap']
