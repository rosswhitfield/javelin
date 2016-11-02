============
Installation
============

Install the latest release
==========================

* Update once we have a release

Requirements
============

* Python 2.7-3.5
* NumPy
* h5py_
* pandas_
* xarray_
* periodictable_

Optional:

* ASE_ (to use the ase atoms structue)

.. _h5py: 
.. _pandas: http://pandas.pydata.org/
.. _xarray: http://xarray.pydata.org
.. _periodictable: http://www.reflectometry.org/danse/elements.html
.. _ASE: https://wiki.fysik.dtu.dk/ase/

Development
===========

A development enviroment can easily be set up with either conda of PyPI

Using Conda
===========

.. code:: sh

   conda env create
   source activate javelin
   python setup.py install

Using PyPI
==========

Tests
=====

The unit tests can be run with pytest_

Install with conda
------------------

.. code:: sh

   conda install pytest

Install with PyPI
-----------------

.. code:: sh

   pip install pytest

.. _pytest: http://pytest.org

