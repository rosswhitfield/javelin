#!/usr/bin/env python
from setuptools import setup, Extension, distutils

if distutils.ccompiler.get_default_compiler() == 'msvc':
    extra_compile_args = ['/openmp']
    extra_link_args = None
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
        ]

setup(
    name='javelin',
    version='0.1.0',
    description='Disordered materials modelling and single crystal diffuse scattering calculator',
    long_description="""Javelin is inspired by DISCUS. It is written in python using modern
    tools, Matplotlib and VTK for plotting, pandas for storing the
    atomic structure and xarray for storing scattering simulations. It
    is designed to play well with other atomic structure analysis
    programs such as ASE and diffpy.

    The scope of javelin is limited to X-ray and neutron single
    crystal nuclear and magnetic diffuse scattering. It will have the
    ability to model disorered structure and refine the structure
    against experimental data.""",
    url='https://github.com/rosswhitfield/javelin',
    author='Ross Whitfield',
    author_email='whitfieldre@ornl.gov',
    license='MIT',
    platforms='any',
    packages=['javelin'],
    classifiers=classifiers,
    ext_modules=[Extension('javelin.fourier_cython', ['javelin/fourier_cython.pyx'],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args)]
)
