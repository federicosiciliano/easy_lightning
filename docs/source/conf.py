# conf.py - Sphinx configuration for Read the Docs

import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath('../../src/easy_lightning'))  # points to the outer easy_lightning
# sys.path.insert(0, os.path.abspath('../../easy_lightning/easy_rec'))  # points to the inner easy_rec
# sys.path.insert(0, os.path.abspath('../../easy_lightning/easy_data'))  # points to the inner easy_data

# sys.path.insert(0, os.path.abspath('../../easy_rec')) # points to the inner easy_data
print("sys.path[0]:", sys.path[0])
print("Contents of sys.path[0]:", os.listdir(sys.path[0]))


# -- Project information -----------------------------------------------------

project = 'Easy Lightning'
copyright = '2025, Federico Siciliano, Filippo Betello, Antonio Purificato, Giulia Di Teodoro, Erica Luciani, Maria Diana Calugaru.'
author = 'Federico Siciliano, Filippo Betello, Antonio Purificato, Giulia Di Teodoro, Erica Luciani, Maria Diana Calugaru.'

release = '0.1.0'
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',        # Automatically document from docstrings
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.githubpages',    # Publish .nojekyll file for GitHub Pages
    'sphinx.ext.todo',           # Support for todo directives
    'sphinx.ext.mathjax',        # Support for LaTeX-style math
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.autosummary',  # <-- Add this

]
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}


# -- Path setup --------------------------------------------------------------

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configurations ------------------------------------------------

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Autodoc
autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Todo
todo_include_todos = True

# Math
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
