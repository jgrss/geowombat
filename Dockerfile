FROM ghcr.io/osgeo/gdal:ubuntu-full-3.9.2
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update -y \
    && apt install -y git build-essential python3-dev

# GeoWombat
RUN useradd -ms /bin/bash wombat 
USER wombat
ENV PATH=/home/wombat/.venv/bin:${PATH}
ENV UV_COMPILE_BYTECODE=1
ARG UV_NO_CACHE=1
ARG CACHEBUST=0
RUN cd \
    && uv venv \
    && . .venv/bin/activate \
    && uv pip install gdal[numpy]==$(gdal-config --version)  \
    && uv pip install git+https://github.com/rdenham/geowombat@meson 
