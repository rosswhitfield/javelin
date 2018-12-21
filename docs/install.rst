============
Installation
============

Install the latest release
==========================

* Update once we have a release

Requirements
============

* Python >= 3.5
* NumPy >= 1.10
* h5py_ >= 2.5
* pandas_ >= 0.17
* xarray_ >= 0.7
* periodictable_ >= 1.4
* cython >= 0.23
* pytables >= 3.2


Optional:

* ASE_ (to use the ase atoms structure) >= 3.14

.. _h5py: 
.. _pandas: http://pandas.pydata.org/
.. _xarray: http://xarray.pydata.org
.. _periodictable: http://www.reflectometry.org/danse/elements.html
.. _ASE: https://wiki.fysik.dtu.dk/ase/

Development
===========

A development environment can easily be set up with either conda or PyPI

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

