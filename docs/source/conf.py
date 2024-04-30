# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
print(pathlib.Path(__file__).parents[2].resolve().as_posix())

import RHM
import RHM.numpy
import RHM.torch
import densemaps
import densemaps.numpy.maps
import densemaps.torch.maps

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ReversibleHarmonicMaps'
copyright = '2024, Robin Magnet'
author = 'Robin Magnet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              "sphinx.ext.napoleon",
              'sphinx_math_dollar',
              'sphinx.ext.mathjax',
              "myst_parser",
              "sphinx_design",
              ]

autodoc_mock_imports = ["sklearn", "densemaps"]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource'}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
