# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Promptify'
copyright = '2023, promptslab'
author = 'promptslab'
release = '2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

suppress_warnings = ["autosectionlabel.*"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['source/_templates']

language = 'English'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "source/_templates/ISSUES_TEMPLATE.rst",
    "TODO/*",
]

# Napoleon settings
napoleon_numpy_docstring = True

# Make sure the target is unique
autosectionlabel_use_sections = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}


source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['source/_static']

# theme
html_theme = 'sphinx_rtd_theme'


# Below html_theme_options config depends on the theme.
html_logo = "../assets/logo-removebg.png"

html_theme_options = {"logo_only": True, "display_version": True}

# -- Options for EPUB output
epub_show_urls = "footnote"