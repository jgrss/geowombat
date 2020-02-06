import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

__version__ = '1.1.3'

pkg_name = 'geowombat'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Geo-enabled n-dimensional arrays from satellite imagery'
git_url = 'http://github.com/jgrss/geowombat.git'
download_url = 'https://github.com/jgrss/geowombat/archive/{VERSION}.tar.gz'.format(VERSION=__version__)
keywords = ['raster', 'remote sensing']

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('requirements.txt') as f:
    required_packages = f.read()


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'data': ['*.png'],
            'geowombat': ['config.ini',
                          'data/*.tif',
                          'data/*.tar.gz',
                          'moving/*.so',
                          'bin/*.tar.gz']}


def get_extensions():

    return [Extension('*',
                      sources=['geowombat/moving/_moving.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp'])]

    # return [Extension('*',
    #                   sources=['geowombat/moving/_moving.pyx'],
    #                   extra_compile_args=['-fopenmp'],
    #                   extra_link_args=['-fopenmp']),
    #         Extension('*',
    #                   sources=['geowombat/radiometry/_fill_gaps.pyx'],
    #                   extra_compile_args=['-fopenmp'],
    #                   extra_link_args=['-fopenmp']),
    #         Extension('*',
    #                   sources=['geowombat/models/_crf.pyx'],
    #                   language='c++')]


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=pkg_name,
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
                    keywords=keywords,
                    url=git_url,
                    download_url=download_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs,
                    classifiers=['Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7'])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
