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
