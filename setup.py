import setuptools
from pathlib import Path
from distutils.core import setup
from distutils.extension import Extension
import re
from collections import defaultdict
import subprocess

try:
    from Cython.Build import cythonize
except:
    raise ImportError('Cython must be installed to build GeoWombat.')

try:
    import numpy as np
except:
    raise ImportError('NumPy must be installed to build GeoWombat.')


# Parse the version from the module.
# Source: https://github.com/mapbox/rasterio/blob/master/setup.py
with open('geowombat/version.py') as f:

    for line in f:

        if line.find("__version__") >= 0:

            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")

            continue

pkg_name = 'geowombat'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Geo-utilities for large-scale processing of air- and space-borne imagery'
git_url = 'https://github.com/jgrss/geowombat'
download_url = 'https://github.com/jgrss/geowombat/archive/{VERSION}.tar.gz'.format(VERSION=version)
keywords = ['raster', 'satellite']
extras = 'extra-requirements.txt'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('requirements.txt') as f:
    required_packages = f.readlines()


# Attempt to get the GDAL binary version
try:

    process = subprocess.Popen(['gdalinfo', '--version'], stdout=subprocess.PIPE, stderr=None)
    gdal_version = str(process.communicate()[0]).split(',')[0].split(' ')[1].strip()

except:
    gdal_version = None

if gdal_version:
    required_packages.append('GDAL=={GDAL_VERSION}\n'.format(GDAL_VERSION=gdal_version))


def get_extra_requires(path, add_all=True):

    with open(path) as fp:

        extra_deps = defaultdict(set)

        for k in fp:

            if k.strip() and not k.startswith('#'):

                tags = set()

                if ':' in k:
                    k, v = k.split(':')
                    tags.update(vv.strip() for vv in v.split(','))

                tags.add(re.split('[<=>]', k)[0])

                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'data': ['*.png'],
            'geowombat': ['config.ini',
                          'data/*.tif',
                          'data/*.TIF',
                          'data/*.gpkg',
                          'data/*.tar.gz',
                          'moving/*.so',
                          'bin/*.tar.gz']}


def get_extensions():

    extensions = [Extension('*',
                            sources=['geowombat/moving/_moving.pyx'],
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-fopenmp'])]

    if Path('geowombat/radiometry/_starfm.pyx').is_file():

        extensions += [Extension('*',
                                 sources=['geowombat/radiometry/_starfm.pyx'],
                                 extra_compile_args=['-fopenmp'],
                                 extra_link_args=['-fopenmp'])]

    return extensions


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=pkg_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=version,
                    long_description=long_description,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_extensions()),
                    zip_safe=False,
                    keywords=' '.join(keywords),
                    url=git_url,
                    download_url=download_url,
                    install_requires=required_packages,
                    extras_require=get_extra_requires(extras),
                    include_dirs=include_dirs,
                    classifiers=['Intended Audience :: Science/Research',
                                 'License :: MIT',
                                 'Topic :: Scientific :: Remote Sensing',
                                 'Programming Language :: Cython',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7',
                                 'Programming Language :: Python :: 3.8'])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
