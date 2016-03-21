#!/usr/bin/env python

from distutils.core import setup

import javelin

setup(
    name=javelin.__name__,
    version=javelin.__version__,
    url=javelin.__url__,
    license=javelin.__license__,
    packages=[
        'javelin'
    ]
)
