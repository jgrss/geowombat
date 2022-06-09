FROM ubuntu:20.04
WORKDIR /source

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:ubuntugis/ppa && \
    apt install \
    python3.8 \
    python3-pip \
    gdal-bin \
    geotiff-bin \
    libgdal-dev \
    libgl1 \
    libgeos++-dev \
    libgeos-3.8.0 \
    libgeos-c1v5 \
    libgeos-dev \
    libgeos-doc \
    libspatialindex-dev \
    git \
    g++ -y

RUN wget https://download.osgeo.org/proj/proj-8.0.0.tar.gz
RUN tar xzvf proj-8.0.0.tar.gz && \
    cd proj-8.0.0 && \
    mkdir build && \
    cd build && \
    ccmake .. -DCMAKE_INSTALL_PREFIX=/usr

RUN export CPLUS_INCLUDE_PATH="/usr/include/gdal"
RUN export C_INCLUDE_PATH="/usr/include/gdal"
RUN export LD_LIBRARY_PATH="/usr/local/lib"

RUN GDAL_VERSION=$(gdal-config --version | awk -F'[.]' '{print $1"."$2"."$3}') && \
    pip install GDAL==$GDAL_VERSION --no-binary=gdal

# pip
RUN pip install -U pip
RUN pip install -U setuptools>=59.5.0 wheel
RUN pip install Cython>=0.29.* numpy>=1.19.0
RUN pip install -U ipython[all]
RUN pip install -U pyshp

# GeoWombat
RUN pip install git+https://github.com/jgrss/geowombat.git#egg=geowombat[coreg,ml,perf,zarr,time,view,web]
RUN pip install jax jaxlib
