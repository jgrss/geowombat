# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))

from datetime import datetime
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
    'numpydoc',
]

autodoc_default_options = {'exclude-members': 'Client, Path'}

# mathjax_path = 'http://cdn.mathjax.org/mathjax/latest/MathJax.js'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'
html_logo = '_static/logo.png'
# html_favicon = ''
pygments_style = 'sphinx'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'page_width': '60%',
    'sidebar_width': '20%',
    'head_font_family': 'Helvetica',
    'font_size': '1.1em',
    'font_family': 'Helvetica',
    'code_font_family': [
        'Consolas',
        'Menlo',
        'DejaVu Sans Mono',
        'Bitstream Vera Sans Mono',
    ],
    'code_font_size': '0.9em',
    'note_bg': '#cccccc',
    'note_border': '#c0c3e2',
    'fixed_sidebar': False,
    'logo': 'logo.png',
    'logo_name': False,
    'github_banner': True,
    'github_button': True,
    'github_user': 'jgrss',
    'github_repo': 'geowombat',
    'repository_url': 'https://github.com/jgrss/geowombat',
    'repository_branch': 'main',
    'use_repository_button': True,
    'use_issues_button': False,
    'home_page_in_toc': False,
    'extra_navbar': '',
    'navbar_footer_text': '',
    'extra_footer': '',
    'anchor': '#d37a7a',
    'anchor_hover_bg': '#d37a7a',
    'anchor_hover_fg': '#d37a7a',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']
ipython_savefig_dir = '_static'

# Disable docstring inheritance
autodoc_inherit_docstrings = False
# autodoc_member_order = 'bysource'
autosummary_generate = True

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
