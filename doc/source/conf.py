# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from datetime import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path

import geowombat as gw

# -- Project information -----------------------------------------------------

project = 'GeoWombat'
copyright = "2020-{:d}, GeoWombat".format(datetime.now().year)
author = ""

# The full version, including alpha/beta/rc tags
release = gw.__version__

# -- General configuration ---------------------------------------------------

# Should special members (like __membername__) and private members
# (like _membername) members be included in the documentation if they
# have docstrings.
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx_automodapi.automodapi',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
    'numpydoc',
    'nbsphinx',
]

# nbsphinx: never execute notebooks during the Sphinx build. Cached
# outputs already saved in the .ipynb files are rendered as-is. To
# refresh outputs, re-execute the notebook locally (e.g. via
# `jupyter nbconvert --to notebook --execute ...`) and commit it.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False

# sphinxcontrib-bibtex configuration (required for version 2.x+)
bibtex_bibfiles = [
    'api/references_vi.bib',
    'api/references_brdf.bib',
    'api/references_tcap.bib',
    'api/references_bandpass.bib',
    'api/references_topo.bib',
    'api/references_radtransforms.bib',
    'api/references_angles.bib',
]

# mathjax_path = 'http://cdn.mathjax.org/mathjax/latest/MathJax.js'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'
html_title = ''
html_extra_path = ['google3b98284c88d0d3f0.html']
logo_only = True
html_logo = '_static/logo.png'
# html_favicon = ''
pygments_style = 'sphinx'

# Theme options. Only keys that ``sphinx_book_theme`` actually accepts;
# many alabaster-theme leftovers (page_width, font_*, anchor_*,
# extra_navbar, github_banner, etc.) were silently dropped and showed
# up as "unsupported theme option" warnings on every build.
html_theme_options = {
    'logo': {
        'alt_text': 'GeoWombat',
    },
    'repository_url': 'https://github.com/jgrss/geowombat',
    'repository_branch': 'main',
    'use_repository_button': True,
    'use_issues_button': False,
    'home_page_in_toc': False,
}

# Prefix `autosectionlabel` targets with the document name, so the same
# section headings (e.g. "Bug fixes", "Classes") across changelog.rst /
# api.rst don't collide and emit hundreds of duplicate-label warnings.
autosectionlabel_prefix_document = True
# Only generate labels for top-level headings. The changelog has dozens
# of releases each with a "Bug fixes" / "New" / "Enhancements"
# subsection, and automodapi adds repeated "Classes" / "Functions"
# subsections under each module — restricting label generation to h1s
# kills ~110 duplicate-label warnings without losing any :ref: target
# that is actually being used.
autosectionlabel_maxdepth = 1

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']
ipython_savefig_dir = '_static'
html_css_files = ['custom.css']

# Disable docstring inheritance
autodoc_inherit_docstrings = False
# autodoc_member_order = 'bysource'
autosummary_generate = True
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '\\usepackage{amsmath}',
}
