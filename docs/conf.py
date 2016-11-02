#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
]

source_suffix = '.rst'

master_doc = 'index'

project = 'Javelin'
copyright = '2016, Ross Whitfield'
author = 'Ross Whitfield'

version = '0.1.0'
release = '0.1.0'

exclude_patterns = ['_build']

pygments_style = 'friendly'

html_theme = 'nature'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Javelindoc'

latex_documents = [
    (master_doc, 'Javelin.tex', 'Javelin Documentation',
     'Ross Whitfield', 'manual'),
]

autodoc_default_flags = ['members', 'undoc-members']
