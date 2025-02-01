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
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.insert(0, os.path.abspath('sphinxext'))


# Setup of mock libraries to override the build fails for C based libraries

#import mock

autodoc_mock_imports = ["numpy", "scipy", "matplotlib", "pandas", "autograd", "assimulo", "cyipopt"]
 
#MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate', 'assimulo']
#for mod_name in MOCK_MODULES:
# sys.modules[mod_name] = mock.Mock()


# -- Project information -----------------------------------------------------

project = 'PharmaPy'
copyright = '2023, Purdue University, Daniel Casas-Orozco, Dan Laky, Inyoung Hur'
author = 'Daniel Casas-Orozco, Dan Laky, Inyoung Hur'

# The full version, including alpha/beta/rc tags
release = '2023'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.doctest', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx', 'sphinxcontrib.bibtex' 
]

bibtex_bibfiles = ['references.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'images/PharmaPy_logo.jpeg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static','images']
html_context = {
    "footer_logos": {
        "row1": [
            {
                "alt": "Purdue University Logo",
                # "src": "Purdue_footer_logo.png",
                "src": "purdue_logo.png",
                "href": "https://www.purdue.edu/",
            },
            {
                "alt": "U.S. Food and Drug Administration Logo",
                "src": "fda_logo.png",
                "href": "https://www.fda.gov/",
            },
        ],
    }
}
