.. _install_conda:

Conda installation
==================

Install dependencies
--------------------

If you already have `GDAL` and `libspatialindex` installed, `geowombat` should work. To test it, see the :ref:`section on testing <test-install>`. If GDAL or libspatialindex are not installed, you will need to do so before having a functional geowombat.

Step-by-step guide
##################

If you are starting from scratch and wish to walk through the full installation procedure, see the following installation steps. We recommend using `virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ and provide an example with `conda <https://docs.conda.io/en/latest/>`_.

1. Create a virtual environment
###############################

Virtual environments with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Conda following the `online instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

Create a virtual Conda environment with a specific Python version::

    conda create --name gwenv python=3.8 cython scipy numpy zarr requests -c conda-forge

Activate the virtual environment::

    conda activate gwenv

Using the virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With an environment activated, the command line should look something like::

    (gwenv)

2. GeoWombat requirements
#########################

`GDAL` (binaries and Python library) and `libspatialindex` can be installed with conda::

    (gwenv) conda install -c conda-forge gdal libspatialindex

3. Install GeoWombat
####################

Install the latest version from GitHub.com::

    (gwenv) pip install git+https://github.com/jgrss/geowombat
