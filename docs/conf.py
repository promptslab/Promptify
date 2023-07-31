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
release = '2.0.2'
version = "latest"

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

# html_context configuration for GitHub edit link
html_context = {
    "display_github": True,
    "github_user": "promptslab",
    "github_repo": "Promptify",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Napoleon settings
napoleon_numpy_docstring = True


source_suffix = [".rst", ".md"]

language = 'English'

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "TODO/*",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# Below html_theme_options config depends on the theme.
html_logo = "../assets/logo-removebg.png"

html_theme_options = {"logo_only": True, "display_version": True}

# -- Options for EPUB output
epub_show_urls = "footnote"
