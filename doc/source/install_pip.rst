.. _install_pip:

PIP installation
================

Install dependencies
--------------------

If you already have `GDAL` and `libspatialindex` installed, `geowombat` should work. To test it, see the :ref:`section on testing <test-install>`. If GDAL or libspatialindex are not installed, you will need to do so before having a functional geowombat.

Step-by-step guide
##################

If you are starting from scratch and wish to walk through the full installation procedure, see the following installation steps. We recommend using `virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ and provide an example with `virtualenv <https://packaging.python.org/key_projects/#virtualenv>`_.

1. Create a virtual environment
###############################

Virtual environments with virtualenv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the `virtualenv` Python package (or use `pyenv <https://github.com/pyenv/pyenv>`_)::

    pip install virtualenv

Create a virtual environment with a specific Python version::

    virtualenv -p python3.8 gwenv

Activate the virtual environment::

    source gwenv/bin/activate

Using the virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With an environment activated, the command line should look something like::

    (gwenv)

2. GeoWombat requirements
#########################

GDAL can be installed via pip, but it requires the GDAL binaries.

Install the GDAL Python bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoWombat will attempt to install the GDAL Python package if the GDAL binaries are installed. However, to install Python GDAL manually, use pip::

    # match the GDAL binaries
    pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --no-binary=gdal
    # which is the same as the following if the GDAL binary version is 2.4.0
    # pip install GDAL==2.4.0 --no-binary=gdal

3. Install GeoWombat
####################

Install the latest version from GitHub.com::

    pip install git+https://github.com/jgrss/geowombat
