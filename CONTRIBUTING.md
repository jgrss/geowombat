# Contibuting to Geowombat
---------------------------

We have two methods for contribution, 1) local install, 2) docker based debugging

1 - Local Install
--------------

## Install GeoWombat

### Clone the latest repo

```commandline
git clone https://github.com/jgrss/geowombat.git
```

### Create a virtual environment

Modify the Python version (3.8.15) as needed

```commandline
pyenv virtualenv 3.8.15 venv.gw
```

### Activate the virtual environment

```commandline
pyenv activate venv.gw
```

### Install pre-commit

```commandline
(venv.gw) pip install pre-commit
(venv.gw) pre-commit install
```

### Install

Install other extras from `setup.cfg` as needed.

```commandline
(venv.gw) cd geowombat/
(venv.gw) pip install -e .[tests]
```

## Create a new branch for your contribution

```commandline
(venv.gw) git checkout -b new_branch_name
```

## After making changes, run tests

```commandline
(venv.gw) cd tests/
(venv.gw) python -m unittest
```

2 - Docker Debuging 
--------------------
Prerequisites
- [Visual Studio Code](https://code.visualstudio.com/download)
- [Docker VS Code Extension](https://code.visualstudio.com/docs/containers/overview)
- [Docker Desktop](https://docs.docker.com/desktop/)

1. Build `geowombat/dockerfiles/gw_docker_debug` by right clicking and hit `Build Image...`
    - This will take a long time the first time only
    - Give the image a name like `gw_debug`, hit Enter
2. Click on dock extension tab on left panel of vscode
3. Under Images click on `gw_debug` right click on `latest`, hit `Run Interactive`
4. Under Individual Containers, right click on your running `gw_debug` instance, hit `Attach Visual Studio Code`
5. Once opened make sure the `python` and `ipython` vscode extensions are installed in your attached vscode server
6. Go to `geowombat/src/debug_script.py` run top cell.
7. Add code and run `debug cell`

View Example Video here:
[![Debug Docker](https://geowombat.readthedocs.io/en/latest/_static/logo.png)](https://youtu.be/hBIE4qmOsgA "Debug Docker")

## Create a Pull Request for the new feature or fix
----------------------

### Commit and push changes

```commandline
git add .
git commit -m 'your commit message'
git push origin new_branch_name
```

### GitHub Pull Request

1. Go to 'Pull Requests' tab
2. Go to 'New Pull Request'
3. Choose 'base:main' and 'compare:new_branch_name'
4. 'Create pull request'
