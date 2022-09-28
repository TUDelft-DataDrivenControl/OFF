# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OFF'
copyright = '2022, Marcus BECKER and Maxime LEJEUNE'
author = 'Marcus BECKER and Maxime LEJEUNE'
release = '0.1'

import sys, os
off_path = f'{os.getcwd().rsplit("/",2)[0]}/03_Code'
sys.path.append(off_path)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Core library for html generation from docstrings
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
