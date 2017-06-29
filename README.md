# Javelin
Javelin is inspired by [DISCUS](http://tproffen.github.io/DiffuseCode). It is written in python using modern tools, [Matplotlib](http://matplotlib.org) and [VTK](http://vtk.org) for plotting, [pandas](http://pandas.pydata.org) for storing the atomic structure and [xarray](http://xarray.pydata.org) for storing scattering simulations. It is designed to play well with other atomic structure analysis programs such as [ASE](https://wiki.fysik.dtu.dk/ase) and [diffpy](http://www.diffpy.org).

The scope of javelin is limited to X-ray and neutron single crystal nuclear and magnetic diffuse scattering.

## Diffuse scattering
While Bragg peaks gives you information on the long-range average diffuse scattering contains a wealth of information from the short-range local structure. Disorder in a material can come in many forms including chemical short-range occupational disorder, displacement disorder, stacking faults and domain structures.


## Installing

```
python setup.py install
```

### Using conda for package management

```
conda env create
source activate javelin
python setup.py install
```

## Running tests

### Unit tests

```
python setup.py test
```
### Doctests

```
py.test --doctest-modules javelin
```

---
[![Documentation Status](https://readthedocs.org/projects/javelin/badge/?version=latest)](http://javelin.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/rosswhitfield/javelin.svg?branch=master)](https://travis-ci.org/rosswhitfield/javelin)
[![Build status](https://ci.appveyor.com/api/projects/status/32ajp5h0qunugdl3?svg=true)](https://ci.appveyor.com/project/rosswhitfield/javelin)
[![codecov.io](https://codecov.io/github/rosswhitfield/javelin/coverage.svg?branch=master)](https://codecov.io/github/rosswhitfield/javelin?branch=master)
[![Code Health](https://landscape.io/github/rosswhitfield/javelin/master/landscape.svg?style=flat)](https://landscape.io/github/rosswhitfield/javelin/master)
