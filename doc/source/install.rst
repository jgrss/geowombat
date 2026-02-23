.. _install:

Installing GeoWombat
====================

Prerequisites
-------------

GeoWombat requires **Python 3.10, 3.11, or 3.12**.

The recommended installation method is **Conda**, which handles all
dependencies automatically on every platform. If you prefer pip, see the
sections below — no system GDAL installation is required.


Install with Conda (recommended)
---------------------------------

`Conda <https://docs.conda.io/en/latest/>`_ provides the easiest and most
consistent installation across Linux, macOS, and Windows. See the
`Conda installation instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
for setup.

Add the ``conda-forge`` channel and install::

    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install geowombat

Or with `mamba <https://mamba.readthedocs.io/>`_ for faster dependency
resolution::

    mamba install geowombat

Alternatively, install in one line following the
`GeoWombat Conda page <https://anaconda.org/conda-forge/geowombat>`_::

    conda install -c conda-forge geowombat


Install with pip
----------------

GeoWombat can be installed from the
`GitHub repository <https://github.com/jgrss/geowombat>`_ using pip. Because
GeoWombat includes Cython extensions that must be compiled, the build requires
a few extra steps compared to a pure-Python package.

.. note::

    **System GDAL is not required.** GeoWombat depends on ``rasterio<1.5.0``,
    which bundles its own GDAL libraries. You do not need to install GDAL
    binaries separately.

.. warning::

    **NumPy 2.x is not yet supported** for building from source. The Cython
    extensions use the NumPy 1.x C API. You must pin ``numpy<2`` in your build
    environment. Additionally, pip's default build isolation pulls in NumPy 2.x,
    so you must use ``--no-build-isolation`` and set ``C_INCLUDE_PATH`` as shown
    below.

.. tabs::

    .. tab:: Linux / macOS

        1. Create a conda environment with build dependencies::

            conda create --name gwenv python=3.11 cython "numpy<2" -y
            conda activate gwenv
            pip install meson-python meson ninja setuptools_scm

        2. Install GeoWombat::

            C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
              pip install --no-build-isolation git+https://github.com/jgrss/geowombat

        To install a specific version (e.g., ``v2.1.22``)::

            C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
              pip install --no-build-isolation git+https://github.com/jgrss/geowombat@v2.1.22

        To include optional extras (e.g., ``ml`` and ``stac``)::

            C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
              pip install --no-build-isolation "geowombat[ml,stac]@git+https://github.com/jgrss/geowombat.git"

    .. tab:: Windows

        On Windows, **Conda is strongly recommended** (see above). If you need
        pip from source, you must first install a C compiler:

        - Install `Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
          with the **"Desktop development with C++"** workload.

        Then, in a conda environment with PowerShell::

            conda create --name gwenv python=3.11 cython "numpy<2" -y
            conda activate gwenv
            pip install meson-python meson ninja setuptools_scm

            $env:C_INCLUDE_PATH = (python -c "import numpy; print(numpy.get_include())")
            pip install --no-build-isolation git+https://github.com/jgrss/geowombat

        To include optional extras::

            $env:C_INCLUDE_PATH = (python -c "import numpy; print(numpy.get_include())")
            pip install --no-build-isolation "geowombat[ml,stac]@git+https://github.com/jgrss/geowombat.git"


Development install
-------------------

For contributors who want an editable install:

.. tabs::

    .. tab:: Setup and install

        Create the environment and install in editable mode::

            conda create --name gwenv python=3.11 cython "numpy<2" -y
            conda activate gwenv
            pip install meson-python meson ninja setuptools_scm

            git clone https://github.com/jgrss/geowombat.git
            cd geowombat/
            C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
              pip install --no-build-isolation -e ".[tests]"

        To include additional extras for development (e.g., ML and docs)::

            C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
              pip install --no-build-isolation -e ".[ml,tests,docs]"

    .. tab:: Rebuild after changes

        After modifying Python or Cython source, rebuild in place::

            C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
              pip install --no-build-isolation -e .

    .. tab:: Code formatting

        Install and activate pre-commit hooks for automatic formatting::

            pip install pre-commit
            pre-commit install

        Run hooks on all files::

            pre-commit run --all-files


Optional extras
---------------

GeoWombat provides optional dependency groups for specific functionality.
Append ``[extra_name]`` to your install command to include them.

.. list-table::
   :header-rows: 1
   :widths: 12 60

   * - Extra
     - Description
   * - ``ml``
     - Machine learning and classification (dask-ml, lightgbm, sklearn-xarray)
   * - ``stac``
     - STAC catalog data access (pystac, stackstac, planetary-computer)
   * - ``coreg``
     - Co-registration (arosics and dependencies)
   * - ``perf``
     - Performance: rtree, pygeos, netCDF4, ray
   * - ``time``
     - Automatic date parsing (dateparser)
   * - ``view``
     - Map visualization (bokeh, descartes, graphviz)
   * - ``web``
     - Web data access (gsutil, wget)
   * - ``zarr``
     - Zarr format support (zarr, numcodecs)
   * - ``docs``
     - Sphinx documentation building
   * - ``tests``
     - Testing dependencies (testfixtures, pre-commit, etc.)

Multiple extras can be combined::

    # Conda (all extras are included automatically)
    conda install geowombat

    # pip (specify extras explicitly)
    C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
      pip install --no-build-isolation "geowombat[ml,stac]@git+https://github.com/jgrss/geowombat.git"


Test the installation
---------------------

Verify the import
~~~~~~~~~~~~~~~~~

If GeoWombat installed correctly, the following should print the version::

    python -c "import geowombat as gw; print(gw.__version__)"

Run the test suite
~~~~~~~~~~~~~~~~~~

Install test dependencies if not already installed::

    pip install testfixtures

Run all tests from the ``tests/`` directory::

    cd geowombat/tests/
    python -m unittest discover -p 'test_*.py'

Run a single test file::

    cd geowombat/tests/
    python -m unittest test_open

Run a single test method::

    cd geowombat/tests/
    python -m unittest test_open.TestOpen.test_open


Troubleshooting
---------------

.. note::

    **Build fails with NumPy errors**

    If you see errors mentioning ``numpy.dtype``, ``PyArray_Descr has no member
    named 'subarray'``, ``NPY_NO_DEPRECATED_API``, or
    ``numpy/_core/include not found``, you likely have NumPy 2.x installed.
    Ensure ``numpy<2`` is installed and that you are using
    ``--no-build-isolation`` with ``C_INCLUDE_PATH`` set::

        pip install "numpy<2"
        C_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())") \
          pip install --no-build-isolation git+https://github.com/jgrss/geowombat

.. note::

    **setuptools_scm version error**

    If the build fails with ``Command 'from setuptools_scm import get_version'
    failed``, install ``setuptools_scm`` in your environment::

        pip install setuptools_scm

.. note::

    **OpenMP not found (macOS)**

    The Cython moving-window extensions optionally use OpenMP for parallelism.
    On macOS, you may need to install ``libomp`` via Homebrew::

        brew install libomp

    The build will succeed without OpenMP, but moving-window operations will
    run single-threaded.
