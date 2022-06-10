## Build a Docker image

1. Clone `geowombat`
```commandline
git clone https://github.com/jgrss/geowombat.git
cd geowombat/
```

2. Build the image

In this command, replace <image name> with the name of the new image.
```commandline
docker build -t <image name> .
```

---
> **NOTE**: Be patient -- the image can take a while to build.
---

3. Run `geowombat` with the new Docker image.
```commandline
docker run -it <image:latest>
```

For example, to run a Jupyter notebook with the Docker image:

First, run:
```commandline
docker run -it -p 8888:8888 geowombat:latest /bin/bash jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
Then, inside the container run:
```commandline
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
