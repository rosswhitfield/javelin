#!/usr/bin/env python
from setuptools import setup, Extension

setup(
    name='javelin',
    version='0.1.0',
    description='',
    url='https://github.com/rosswhitfield/javelin',
    author='Ross Whitfield',
    author_email='whitfieldre@ornl.gov',
    license='MIT',
    packages=['javelin'],
    ext_modules=[Extension('javelin.fourier_cython', ['javelin/fourier_cython.pyx'],
                           extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]
)
