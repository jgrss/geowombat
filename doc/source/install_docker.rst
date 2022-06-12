.. _install_docker:

Docker
======

1. Build a local image
######################

Navigate to the `geowombat/dockerfiles` directory and build::

    git clone https://github.com/jgrss/geowombat.git
    docker build -t <your image name> .

Check that it was built::

    docker image ls

2. Use the local Docker image to run `geowombat`
################################################

Enter the image::

    docker run -it <your image name> bash

