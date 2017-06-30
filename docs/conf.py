#!/usr/bin/env python3
# -*- coding: utf-8 -*-

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode'
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
                       'xarray': ('http://xarray.pydata.org/en/stable/', None),
                       'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
                       'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
                       'diffpy.Structure': ('http://www.diffpy.org/diffpy.structure/', None)}

autodoc_default_flags = ['members', 'undoc-members']
