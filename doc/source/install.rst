.. _install:

Installing GeoWombat
====================

Install with pip
----------------

If your system has GDAL installed then `geowombat <https://gdal.org/>`_ can be installed via `pip`
directly from the GitHub repository.

.. tabs::

    .. tab:: pip (latest version)

        Install directly from the GitHub.com repository::

            pip install git+https://github.com/jgrss/geowombat

    .. tab:: pip (specific version)

        Specify a version to install (e.g., `geowombat==2.0.10`)::

            pip install git+https://github.com/jgrss/geowombat@v2.0.10

    .. tab:: pip (Clone repository)

        .. tabs::

            .. tab:: Clone and install

                Clone the repository and build locally (requires `git`)::

                    cd clone_dir/
                    git clone https://github.com/jgrss/geowombat.git
                    cd geowombat/
                    pip install .

            .. tab:: Pull and update

                Pull the latest and rebuild::

                    cd clone_dir/geowombat/
                    git pull origin main
                    pip install -U .

            .. tab:: Branch and develop

                Create a new branch and install as editable::

                    cd clone_dir/geowombat/
                    git checkout -b new_branch_name
                    pip install -e .

Install with Conda
------------------

`Conda <https://docs.conda.io/en/latest/>`_ provides an easy and consistent installation method regardless of
operating system. See the `conda installation instructions <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
for setup.

Installing `geowombat` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Once the `conda-forge` channel has been enabled, `geowombat` can be installed with `conda`::

    conda install geowombat

or with `mamba`::

    mamba install geowombat


Detailed install
----------------

The GDAL binaries must be installed prior to installing GeoWombat. Below are instructions on how to build GDAL for your
operating system.

.. tabs::

    .. tab:: Full build

        .. tabs::

            .. tab:: 1 - GDAL

                .. tabs::

                    .. tab:: OSX

                        Install `gcc` to compile and install `GDAL <https://gdal.org/>`_. On OSX, these are easiest to
                        install via `homebrew <https://docs.brew.sh/Installation>`_.

                        From the terminal window, update `brew` and install::

                            brew update
                            brew upgrade
                            brew install gcc
                            brew install gdal openssl spatialindex

                    .. tab:: Linux

                        Install requirements on Linux using `apt`::

                            apt update -y && apt upgrade -y && \
                            apt install -y software-properties-common && \
                            add-apt-repository ppa:ubuntugis/ppa && \
                            apt update -y && apt install -y \
                            gdal-bin \
                            geotiff-bin \
                            git \
                            libgdal-dev \
                            libgl1 \
                            libspatialindex-dev \
                            wget \
                            python3 \
                            python3-pip \
                            pip \
                            g++

                    .. tab:: Windows

                        Using `conda`, install GDAL by::

                            conda install -c conda-forge gdal

                        For more details and possibly other options (e.g., .exe), refer to this
                        `GDAL on Windows blog <https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/>`_.

            .. tab:: 2 - Post-GDAL

                After GDAL has been installed, ensure that the binaries are in the system path by::

                    gdalinfo --version

                which should printout something like::

                    GDAL 3.3.2, released 2021/09/01

                Note that the version can also be obtained by::

                    gdal-config --version

            .. tab:: 3 - Virtual environment

                .. tabs::

                    .. tab:: Virtual environments

                        Python virtual environments are not required, but are good practice. There are various packages available
                        that can be used to create a virtual environment. For example, the built-in
                        `venv <https://docs.python.org/3/library/venv.html>`_, can be used like::

                            python -m venv <path to virtual environment>

                        The `virtualenv package <https://virtualenv.pypa.io/en/latest/>`_ can be installed from `PyPI <https://pypi.org/>`_::

                            pip install virtualenv

                        The `pyenv package <https://github.com/pyenv/pyenv>`_ is another good option.

                        **Creating a virtual environment**

                        Create a virtual environment with a specific Python version using `virtualenv`::

                            virtualenv -p python3.8 gwenv

                        Activate the virtual environment::

                            source gwenv/bin/activate

                    .. tab:: Virtual environments with Conda

                        Virtual environments can also be created using `conda`. First, install `conda`
                        following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

                        Create a virtual Conda environment with a specific Python version::

                            conda create --name gwenv python=3.8 cython numpy

                        Activate the virtual environment::

                            conda activate gwenv

                        Install geowombat requirements via conda-forge::

                            conda config --env --add channels conda-forge
                            conda config --env --set channel_priority strict

                    .. tab:: Using a virtual environment

                        With a virtual environment activated, the command line should look something like::

                            (gwenv)

                        where `gwenv` is the name of your virtual environment. Once activated, all subsequent
                        Python package installations will be isolated to this environment.

            .. tab:: 4 - Python GDAL

                The Python GDAL package version must match the GDAL binaries version. For this reason, GeoWombat
                does not attempt to install the GDAL Python package. Be sure to use the same version printed from::

                    gdalinfo --version

                or::

                    gdal-config --version

                For example, if the version from the above commands is `3.3.2` then install the Python GDAL API by::

                    (gwenv) pip install GDAL==3.3.2

                .. note::

                    In Windows we recommend the use of `conda` since `pip` often requires the use of precompiled
                    binaries, which can get tricky. If using `pip`, there may be some cases where installing packages
                    will not be successful in Windows. In these cases please refer to the precompiled wheel files at
                    `Christoph Gohlke's website <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

            .. tab:: 5 - GeoWombat

                Install the latest version from GitHub.com::

                    (gwenv) pip install git+https://github.com/jgrss/geowombat

            .. tab:: 6 - Updating

                To update GeoWombat::

                    (gwenv) pip install --upgrade git+https://github.com/jgrss/geowombat

            .. tab:: 7 - Optional extras

                GeoWombat has a lot of additional capabilities, some of which you may or may not want to use.
                For this reason, we allow the user to decide which dependencies they want to install.

                Install GeoWombat with libraries for building Sphinx docs::

                    (gwenv) pip install "geowombat[docs]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries for co-registration::

                    (gwenv) pip install arosics --no-deps && pip install "geowombat[coreg]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries for machine learning and classification::

                    (gwenv) pip install "geowombat[ml]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries for pygeos, netcdf and ray support::

                    (gwenv) pip install "geowombat[perf]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries for parsing dates automatically::

                    (gwenv) pip install "geowombat[time]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries with map-making dependencies::

                    (gwenv) pip install "geowombat[view]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries for accessing hosted data::

                    (gwenv) pip install "geowombat[web]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with libraries for streaming data from STAC::

                    (gwenv) pip install "geowombat[stac]@git+https://github.com/jgrss/geowombat.git"

                Multiple extras can be included::

                    (gwenv) pip install "geowombat[perf,stac]@git+https://github.com/jgrss/geowombat.git"

                Install GeoWombat with all extra libraries::

                    (gwenv) pip install "geowombat[all]@git+https://github.com/jgrss/geowombat.git"

    .. tab:: Docker

        .. tabs::

            .. tab:: Build from pre-built image

                A pre-built Docker image is available as `mmann1123/gw_pygis` on `Docker Hub <https://hub.docker.com/>`_.
                To use this image, follow the Docker build instructions at
                `PyGIS <https://pygis.io/docs/b_conda_started.html#docker-for-spatial-python-gdal-included>`_.

            .. tab:: Build image from scratch

                If you want to build an image from scratch, a Dockerfile is provided in GeoWombat. Using this file,
                a Docker image can be built by::

                    git clone https://github.com/jgrss/geowombat.git
                    cd geowombat/
                    docker build -t <your image name> .

                Enter the image by::

                    docker run -it <your image name> bash

Test the installation
---------------------

Test the import
###############

If GeoWombat installed correctly, you should be able to run the following command from the terminal::

    python -c "import geowombat as gw;print(gw.__version__)"

or in Python:

.. ipython:: python

    import geowombat as gw
    print(gw.__version__)

Unittests
#########

Install `testfixtures` (used to test logging outputs in `test_config.py`)::

    pip install testfixtures

Run all unittests inside GeoWombat's `/tests` directory::

    cd geowombat/tests
    python -m unittest

Run an individual test::

    python test_open.py


Installation Notes
~~~~~~~~~~~~~~~~~~

.. note::

    **GDAL install:**
    GeoWombat requires `GDAL <https://gdal.org/>`_ and `libspatialindex <https://libspatialindex.org/>`_.

    This GDAL requirement is a prerequisite itself for the `Python GDAL bindings <https://pypi.org/project/GDAL/>`_.

.. note::

    **GDAL Paths in Linux:**
    Although we think this is now resolved, earlier GDAL installations had issues with paths. If this is the case, try
    updating the environment paths::

        export CPLUS_INCLUDE_PATH=/usr/include/gdal
        export C_INCLUDE_PATH=/usr/include/gdal
        export LD_LIBRARY_PATH=/usr/local/lib

        # Optional, add permanently to a .profile file
        # echo 'export CPLUS_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
        # echo 'export C_INCLUDE_PATH="/usr/include/gdal"' >> ~/.profile
        # echo 'export LD_LIBRARY_PATH="/usr/local/lib"' >> ~/.profile

.. note::

    **EPSG File Missing in Linux:**
    If you install GDAL 3.x on Ubuntu, when importing `rasterio` you may receive an error saying

    - `CPLE_OpenFailedError: Unable to open EPSG support file gcs.csv`.

    This error is documented on rasterio's `ReadTheDocs page <https://rasterio.readthedocs.io/en/latest/faq.html>`_
    and `GitHub page <https://github.com/mapbox/rasterio/issues/1787>`_. If the suggested solutions do not fix the
    issue, you can try setting the `GDAL_DATA` environment variable to point to Fiona (which will be installed
    automatically when installing GeoWombat). For example, if you have setup a virtual environment, the `GDAL_DATA`
    variable can point to `/path/to/myenv/lib/python3.8/site-packages/fiona/gdal_data`, where `/path/to/myenv` is
    the name of your virtual environment path directory. Change 3.8 if using a different Python version.
