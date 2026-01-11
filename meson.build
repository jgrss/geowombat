project(
    'geowombat',
    ['c', 'cython'],
    version: run_command('python3', '-c', 'from setuptools_scm import get_version; print(get_version())', check: true).stdout().strip()
)

py_mod = import('python')
py3 = py_mod.find_installation('python3')

# Root package files
py3.install_sources(
    [
        'src/geowombat/__init__.py',
        'src/geowombat/config.py',
        'src/geowombat/config.ini',
        'src/geowombat/handler.py',
    ],
    subdir: 'geowombat',
    pure: false
)

# backends
py3.install_sources(
    [
        'src/geowombat/backends/__init__.py',
        'src/geowombat/backends/dask_.py',
        'src/geowombat/backends/gdal_.py',
        'src/geowombat/backends/rasterio_.py',
        'src/geowombat/backends/xarray_.py',
        'src/geowombat/backends/xarray_rasterio_.py',
        'src/geowombat/backends/zarr_.py',
    ],
    subdir: 'geowombat/backends',
    pure: false
)

# bin
py3.install_sources(
    [
        'src/geowombat/bin/__init__.py',
        'src/geowombat/bin/ESPA.tar.gz',
    ],
    subdir: 'geowombat/bin',
    pure: false
)

# core
py3.install_sources(
    [
        'src/geowombat/core/__init__.py',
        'src/geowombat/core/api.py',
        'src/geowombat/core/base.py',
        'src/geowombat/core/conversion.py',
        'src/geowombat/core/geoxarray.py',
        'src/geowombat/core/io.py',
        'src/geowombat/core/parallel.py',
        'src/geowombat/core/properties.py',
        'src/geowombat/core/series.py',
        'src/geowombat/core/sops.py',
        'src/geowombat/core/stac.py',
        'src/geowombat/core/util.py',
        'src/geowombat/core/vi.py',
        'src/geowombat/core/windows.py',
    ],
    subdir: 'geowombat/core',
    pure: false
)

# data
py3.install_sources(
    [
        'src/geowombat/data/__init__.py',
        'src/geowombat/data/L3B_S2B_00390821JXN0L2A_20210319_20220730_C01.nc',
        'src/geowombat/data/l8_224077_20200518_B2_nan.tif',
        'src/geowombat/data/l8_224078_20200518_B2_nan.tif',
        'src/geowombat/data/LC08_L1TP_224077_20200518_20200518_01_RT_B2_60m.TIF',
        'src/geowombat/data/LC08_L1TP_224077_20200518_20200518_01_RT_B2.TIF',
        'src/geowombat/data/LC08_L1TP_224077_20200518_20200518_01_RT_B3.TIF',
        'src/geowombat/data/LC08_L1TP_224077_20200518_20200518_01_RT_B4.TIF',
        'src/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT_B2.TIF',
        'src/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT_B3.TIF',
        'src/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT_B4.TIF',
        'src/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT_points.gpkg',
        'src/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT_polygons.gpkg',
        'src/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT.TIF',
        'src/geowombat/data/LC08_L2SP_224078_20200127_02_T1_LC08_L2SP_224078_20200127_20200823_02_T1_MTL.txt',
        'src/geowombat/data/LE07_L2SP_225078_20110306_02_T1_B1.tif',
        'src/geowombat/data/LE07_L2SP_225078_20110306_02_T1_B1.tif.aux.xml',
        'src/geowombat/data/LE07_L2SP_225078_20110306_02_T1_LE07_L2SP_225078_20110306_20200910_02_T1_ANG.txt',
        'src/geowombat/data/LE07_L2SP_225078_20110306_02_T1_SZA.tif',
        'src/geowombat/data/mgrs.tar.gz',
        'src/geowombat/data/rgbn_20160101.tif',
        'src/geowombat/data/rgbn_20160401.tif',
        'src/geowombat/data/rgbn_20160517.tif',
        'src/geowombat/data/rgbn_20170203.tif',
        'src/geowombat/data/rgbn_suba.tif',
        'src/geowombat/data/rgbn_subb.tif',
        'src/geowombat/data/rgbn.tif',
        'src/geowombat/data/srtm30m_bounding_boxes.gpkg',
        'src/geowombat/data/wrs2.tar.gz',
    ],
    subdir: 'geowombat/data',
    pure: false
)

# ml
py3.install_sources(
    [
        'src/geowombat/ml/__init__.py',
        'src/geowombat/ml/classifiers.py',
        'src/geowombat/ml/transformers.py',
    ],
    subdir: 'geowombat/ml',
    pure: false
)

# moving (Python files only, Cython handled in subdir)
py3.install_sources(
    [
        'src/geowombat/moving/__init__.py',
    ],
    subdir: 'geowombat/moving',
    pure: false
)

# radiometry (Python files only, Cython handled in subdir)
py3.install_sources(
    [
        'src/geowombat/radiometry/__init__.py',
        'src/geowombat/radiometry/angles.py',
        'src/geowombat/radiometry/brdf.py',
        'src/geowombat/radiometry/mask.py',
        'src/geowombat/radiometry/qa.py',
        'src/geowombat/radiometry/sixs.py',
        'src/geowombat/radiometry/sr.py',
        'src/geowombat/radiometry/topo.py',
    ],
    subdir: 'geowombat/radiometry',
    pure: false
)

# tasks
py3.install_sources(
    [
        'src/geowombat/tasks/__init__.py',
        'src/geowombat/tasks/pipeline.py',
    ],
    subdir: 'geowombat/tasks',
    pure: false
)

# util
py3.install_sources(
    [
        'src/geowombat/util/__init__.py',
        'src/geowombat/util/percentiles.pxd',
        'src/geowombat/util/plotting.py',
        'src/geowombat/util/web.py',
    ],
    subdir: 'geowombat/util',
    pure: false
)

# Cython extension subdirectories
subdir('src/geowombat/moving')
subdir('src/geowombat/radiometry')
