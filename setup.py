import platform
from pathlib import Path
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
except:
    raise ImportError('Cython must be installed to build GeoWombat.')

try:
    import numpy as np
except:
    raise ImportError('NumPy must be installed to build GeoWombat.')


compile_args = ['-fopenmp']
link_args = ['-fopenmp']

if platform.system().lower() == 'darwin':
    compile_args.insert(0, '-Xpreprocessor')
    link_args = ['-lomp']


def get_extensions():
    extensions = [Extension(
        '*',
        sources=['src/geowombat/moving/_moving.pyx'],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )]

    if Path('src/geowombat/moving/_test.pyx').is_file():
        extensions += [Extension(
            '*',
            sources=['src/geowombat/moving/_test.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )]

    if Path('src/geowombat/radiometry/_fusion.pyx').is_file():
        extensions += [Extension(
            '*',
            sources=['src/geowombat/radiometry/_fusion.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )]

    return extensions


def setup_package():
    metadata = dict(
        ext_modules=cythonize(get_extensions()),
        include_dirs=[np.get_include()]
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
