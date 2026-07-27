.. _install:

Installing GeoWombat
====================

Prerequisites
-------------

GeoWombat requires **Python 3.10, 3.11, 3.12, or 3.13**.

The recommended installation method is **Conda**, which handles the core
dependencies (including GDAL/rasterio) automatically on every platform.
Optional extras are available as separate Conda sub-packages (see
:ref:`optional-extras` below). If you prefer pip, see the sections below.


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

GeoWombat is published on `PyPI <https://pypi.org/project/geowombat/>`_ and can
be installed with pip::

    pip install geowombat

To include optional extras (e.g., ``ml`` and ``stac``)::

    pip install "geowombat[ml,stac]"

To install everything needed to run the example notebooks in one step::

    pip install "geowombat[all]"

.. note::

    GeoWombat is currently distributed on PyPI as a source distribution (there
    are no pre-built wheels yet), so pip compiles its Cython extensions at
    install time. This requires a C compiler (``gcc`` on Linux, Xcode
    command-line tools on macOS, or Visual Studio Build Tools on Windows). On
    Windows, **Conda is strongly recommended** to avoid setting up a compiler.

Installing from source (GitHub) is also supported, and is required when you want
an unreleased version or a specific commit. The same C compiler requirement
applies.

.. tabs::

    .. tab:: Linux / macOS

        Install GeoWombat from GitHub::

            pip install git+https://github.com/jgrss/geowombat

        To install a specific version (e.g., ``v2.1.22``)::

            pip install git+https://github.com/jgrss/geowombat@v2.1.22

        To include optional extras (e.g., ``ml`` and ``stac``)::

            pip install "geowombat[ml,stac]@git+https://github.com/jgrss/geowombat.git"

    .. tab:: Windows

        On Windows, **Conda is strongly recommended** (see above). If you need
        pip from source, you must first install a C compiler:

        - Install `Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
          with the **"Desktop development with C++"** workload.

        Then install::

            pip install git+https://github.com/jgrss/geowombat

        To include optional extras::

            pip install "geowombat[ml,stac]@git+https://github.com/jgrss/geowombat.git"


GDAL system libraries
---------------------

**Most users do not need to install GDAL separately.** GeoWombat depends on
``rasterio<1.5.0``, which ships with its own bundled GDAL libraries on all
platforms. The bundled GDAL is sufficient for reading and writing GeoTIFFs,
reprojection, warping, and all standard GeoWombat operations.

However, a **system GDAL installation** may be needed if you:

- Want to use GDAL command-line tools (``gdalinfo``, ``gdal_translate``, etc.)
  outside of Python
- Need drivers not included in the rasterio wheel (e.g., certain HDF or
  database-backed formats)
- Are building rasterio or GDAL Python bindings from source

.. tabs::

    .. tab:: macOS

        Install GDAL via `Homebrew <https://brew.sh/>`_::

            brew install gdal

        Verify the installation::

            gdalinfo --version

    .. tab:: Windows

        The easiest option is to use **Conda**, which installs GDAL
        automatically as a dependency of rasterio::

            conda install -c conda-forge gdal

        Alternatively, install via `OSGeo4W <https://trac.osgeo.org/osgeo4w/>`_:

        1. Download the `OSGeo4W installer <https://trac.osgeo.org/osgeo4w/>`_.
        2. Run the installer and select **GDAL** from the package list.
        3. Add the OSGeo4W ``bin`` directory to your system ``PATH``.

        Verify::

            gdalinfo --version

    .. tab:: Linux

        Install from your package manager::

            # Ubuntu / Debian
            sudo apt install gdal-bin libgdal-dev

            # Fedora / RHEL
            sudo dnf install gdal gdal-devel

        Or via Conda::

            conda install -c conda-forge gdal

        Verify::

            gdalinfo --version


Development install
-------------------

For contributors who want an editable install:

.. warning::

    **Editable installs require** ``--no-build-isolation``. Meson-python
    caches the NumPy include path at configure time. With build isolation,
    this path points to a temporary directory that is deleted after install,
    causing ``import geowombat`` to fail on the next rebuild. Using
    ``--no-build-isolation`` ensures paths point to your actual environment.

.. tabs::

    .. tab:: Setup and install

        Clone and install in editable mode::

            git clone https://github.com/jgrss/geowombat.git
            cd geowombat/
            pip install --no-build-isolation -e ".[tests]"

        To include additional extras for development (e.g., ML and docs)::

            pip install --no-build-isolation -e ".[ml,tests,docs]"

    .. tab:: Rebuild after changes

        After modifying Python or Cython source, rebuild in place::

            pip install --no-build-isolation -e .

    .. tab:: Code formatting

        Install and activate pre-commit hooks for automatic formatting::

            pip install pre-commit
            pre-commit install

        Run hooks on all files::

            pre-commit run --all-files


.. _optional-extras:

Optional extras
---------------

GeoWombat provides optional dependency groups for specific functionality.

.. list-table::
   :header-rows: 1
   :widths: 12 60

   * - Extra
     - Description
   * - ``ml``
     - Machine learning and classification (dask-ml, lightgbm, sklearn-xarray, numba)
   * - ``dl``
     - Deep learning (PyTorch, pytorch-tabnet, torchgeo, segmentation-models-pytorch)
   * - ``detect``
     - Object detection (ultralytics/YOLO, torchmetrics, pycocotools, pillow, opencv-python)
   * - ``sam``
     - Segment Anything Model (Meta's segment-anything)
   * - ``stac``
     - STAC catalog data access (pystac, stackstac, planetary-computer)
   * - ``coreg``
     - Sub-pixel co-registration (arosics and dependencies)
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

Installing extras with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

Append ``[extra_name]`` to your pip install command. Multiple extras can be
combined::

    pip install "geowombat[ml,stac]@git+https://github.com/jgrss/geowombat.git"
    pip install "geowombat[ml,detect]@git+https://github.com/jgrss/geowombat.git"

Installing extras with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Conda, optional extras are installed as **separate sub-packages** — they
are *not* included by a plain ``conda install geowombat``. Install them by
name, e.g.::

    conda install geowombat-ml geowombat-stac

The available Conda sub-packages are: ``geowombat-ml``, ``geowombat-stac``,
``geowombat-coreg``, ``geowombat-perf``, ``geowombat-view``, ``geowombat-web``,
``geowombat-zarr``, ``geowombat-docs``, ``geowombat-tests``, ``geowombat-dl``,
``geowombat-detect``, and ``geowombat-sam``.

See the `conda-forge feedstock <https://github.com/conda-forge/geowombat-feedstock>`_
for the full recipe.

.. note::

    **Conda differences from pip extras:**

    - ``time`` (dateparser) is included in the **base** Conda package
      automatically; with pip it is an optional extra.
    - ``geowombat-dl``: the Conda sub-package ships ``pytorch``, ``torchgeo``,
      and ``segmentation-models-pytorch`` but **omits** ``pytorch-tabnet``.
      The current ``pytorch-tabnet`` build on conda-forge pins
      ``pytorch <2.0``, which conflicts with ``torchgeo`` and
      ``segmentation-models-pytorch``. If you need TabNet, install it with
      pip into the same env after the Conda install::

          conda install geowombat-dl
          pip install pytorch-tabnet

    - ``geowombat-detect``: ``opencv-python`` (``py-opencv`` on conda-forge)
      is left unpinned; the solver matches it to the installed NumPy.
    - ``geowombat-sam``: ``segment-anything`` (the upstream PyPI package) is
      not available on conda-forge. The ``geowombat-sam`` sub-package is a
      placeholder that only pins the base ``geowombat``; install
      ``segment-anything`` with pip after the Conda install::

          conda install geowombat-sam
          pip install segment-anything

    - A few packages are unavailable on conda-forge (or unavailable in a
      compatible build) and must be installed via pip when using Conda:
      ``wrapt-timeout-decorator`` (stac), ``sphinx_tabs`` (docs), ``pygeos``
      (perf; deprecated — its functionality is now in shapely 2.0+),
      ``segment-anything`` (sam; install via pip as described above), and
      ``pytorch-tabnet`` (dl, conda only — works fine on pip).


.. _example-notebooks:

Running the example notebooks
-----------------------------

The example notebooks in the documentation each depend on one or more optional
extras. Every notebook lists its exact ``pip`` and ``conda`` install commands in
a **Requirements** cell at the top, so install the extra(s) for the notebook you
want to run. To cover every notebook at once, install all extras with
``pip install "geowombat[all]"`` (or the matching ``geowombat-*`` sub-packages
with Conda; see :ref:`optional-extras`).

.. note::

   Conda users installing the ``stac`` sub-package must also
   ``pip install wrapt-timeout-decorator`` (not on conda-forge). ``geowombat-dl``
   omits ``pytorch-tabnet`` (``pip install pytorch-tabnet`` if you need TabNet),
   and the ``object_detection`` notebook's SAM-refiner section additionally needs
   ``pip install segment-anything``. See :ref:`optional-extras` for details.


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

    **NumPy 1.x and 2.x are both supported**

    GeoWombat builds its compiled extensions against NumPy 2.x, which keeps
    them ABI-compatible with NumPy >=1.23 at runtime, so it runs under either
    NumPy 1.x or 2.x. The runtime requirement is ``numpy>=1.23,<3``.

    If you **switch** the NumPy major version of an existing **editable**
    install (e.g. ``pip install "numpy<2"`` on top of an editable build that
    was compiled against NumPy 2), rebuild the extensions so their ABI matches
    the active NumPy::

        rm -rf build/
        pip install --no-build-isolation -e ".[tests]"

    If you see build errors mentioning ``PyArray_Descr has no member named
    'subarray'`` or ``numpy/_core/include not found``, you are building against
    a NumPy older than 2.0 with a modern Cython. Upgrade pip and clear the
    build cache::

        pip install --upgrade pip
        pip install --no-cache-dir git+https://github.com/jgrss/geowombat

.. note::

    **Editable install fails with** ``re-building the geowombat meson-python editable wheel package failed``

    This means the meson build cache has stale paths (usually from a
    previous build isolation environment). Delete the build directory and
    reinstall::

        rm -rf build/
        pip install --no-build-isolation -e ".[tests]"

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
