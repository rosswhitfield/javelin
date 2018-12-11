#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive'
]

source_suffix = '.rst'

master_doc = 'index'

project = 'Javelin'
copyright = '2017, Ross Whitfield'
author = 'Ross Whitfield'

version = '0.1.0'
release = '0.1.0'

exclude_patterns = ['_build']

pygments_style = 'friendly'

html_theme = 'sphinx_rtd_theme'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Javelindoc'

latex_documents = [
    (master_doc, 'Javelin.tex', 'Javelin Documentation',
     'Ross Whitfield', 'manual'),
]

intersphinx_mapping = {'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'xarray': ('https://xarray.pydata.org/en/stable/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
                       'diffpy.Structure': ('https://www.diffpy.org/diffpy.structure/', None)}

autodoc_default_flags = ['members', 'undoc-members']

# Use legacy numpy printing. This fix is made to keep doctests functional.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass
