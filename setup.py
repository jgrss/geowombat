import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

__version__ = '1.0.0a'

mappy_name = 'GeoWombat'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Geo-enabled n-dimensional arrays from satellite imagery'
git_url = 'http://github.com/jgrss/geowombat.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

required_packages = ['cython',
                     'matplotlib',
                     'GDAL',
                     'pandas',
                     'geopandas',
                     'numpy',
                     'rasterio',
                     'xarray',
                     'dask',
                     'dask-ml',
                     'dask[dataframe]',
                     'distributed',
                     'affine',
                     'joblib',
                     'tqdm',
                     'shapely',
                     'h5netcdf',
                     'netCDF4']


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'data': ['*.png'],
            'geowombat': ['config.ini',
                          'data/*.tif',
                          'moving/*.so']}


def get_extensions():

    return [Extension('*',
                      sources=['geowombat/moving/_moving.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp'])]


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=mappy_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=__version__,
                    long_description=long_description,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_extensions()),
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
