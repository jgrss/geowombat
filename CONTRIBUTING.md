# Contributing to GeoWombat

Contributions are welcome! The steps below cover setting up a local development
environment, running the tests, and opening a pull request.

## How to Contribute

- **Bug reports** — open an [issue](https://github.com/jgrss/geowombat/issues)
  with a minimal reproducible example, the traceback, and your GeoWombat,
  Python, and OS versions.
- **Feature requests** — open an issue describing the use case before starting
  work, so the approach can be discussed.
- **Documentation** — fixes and additions to the docs (`doc/`) and docstrings
  are always welcome and follow the same pull-request flow as code.
- **Code** — bug fixes and features via pull requests (see below). For large
  changes, please open an issue first.

## AI-Assisted Contributions

AI tools (e.g. GitHub Copilot, Claude, ChatGPT) are welcome, but the same bar
applies as to any other contribution:

- **You are responsible for your contribution.** Review, understand, and test
  any AI-generated code before submitting it — please do not open pull requests
  containing code you cannot explain.
- **Follow the project conventions** below (formatting, tests, Conventional
  Commits). AI-generated output must pass the same pre-commit hooks and CI.
- **Disclose significant AI assistance**, e.g. with an `Assisted-by:` or
  `Co-authored-by:` commit trailer, so review and attribution stay transparent.
- **Only contribute code you have the right to submit** under the project's MIT
  license.

For a fuller policy template, see <https://aipolicy.1mb.dev/>.

## Local Install

### Install GeoWombat

#### Clone the latest repo

```commandline
git clone https://github.com/jgrss/geowombat.git
```

#### Create a virtual environment

Modify the Python version (i.e., 3.11) as needed. GeoWombat is tested on
Python 3.10–3.12.

```commandline
pyenv virtualenv 3.11 venv.gw
```

#### Activate the virtual environment

```commandline
pyenv activate venv.gw
```

#### Install pre-commit

```commandline
(venv.gw) pip install pre-commit
(venv.gw) pre-commit install
```

#### Install GeoWombat

Install other extras from `pyproject.toml` as needed.

```commandline
(venv.gw) cd geowombat/
(venv.gw) pip install -e .[tests]
```

### Create a new branch for your contribution

```commandline
(venv.gw) git checkout -b new_branch_name
```

### After making changes, run tests

```commandline
(venv.gw) cd tests/
(venv.gw) python -m unittest
```

## Coding Conventions

Formatting and linting are enforced by [pre-commit](https://pre-commit.com/)
(installed above). The hooks run automatically on `git commit`, or manually
with `pre-commit run --all-files`:

- **[black](https://black.readthedocs.io/)** — code formatting (with
  `--skip-string-normalization`).
- **[isort](https://pycqa.github.io/isort/)** — import sorting.
- **[flake8](https://flake8.pycqa.org/)** — linting.
- **[docformatter](https://github.com/PyCQA/docformatter)** — docstring
  formatting, wrapped at 79 characters.

Additional conventions:

- **Docstrings** — Google style (rendered by Sphinx napoleon).
- **Commit messages** — [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `style:`, `test:`, `chore:`).
  These drive automated versioning and the changelog, so write descriptive
  messages — `feat:` and `fix:`/`perf:` trigger minor and patch releases.
- **Tests** — add or update tests under `tests/` for any behavior change.

## Create a Pull Request

#### Commit and push changes

```commandline
git add .
git commit -m 'fix: short description of the change'
git push origin new_branch_name
```

#### GitHub Pull Request

1. Go to the 'Pull Requests' tab
2. Click 'New Pull Request'
3. Choose 'base:main' and 'compare:new_branch_name'
4. Click 'Create pull request'

CI (tests across Python 3.10–3.12) and a non-required documentation build run
on every pull request; please make sure they pass.

## Releases (for maintainers)

Releases are automated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/) via
the `Release` workflow, which runs after CI succeeds on `main`:

1. The next version is derived from the Conventional Commit messages since the
   last tag; `CHANGELOG.md` is updated and a `chore(release): vX.Y.Z` commit and
   tag are pushed.
2. The new tag triggers `publish.yml`, which builds the sdist and publishes to
   PyPI, and `docs.yml`, which triggers a ReadTheDocs build.

Notes:

- Because versioning is commit-driven, a release is a no-op unless a
  `feat`/`fix`/`perf` commit has landed. Nothing to tag by hand.
- The build backend is [meson-python](https://meson-python.readthedocs.io/);
  Cython extensions are compiled at build time, so `ninja` and a C/C++ compiler
  must be available in build and CI environments.
