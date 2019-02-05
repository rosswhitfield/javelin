Working with DISCUS
===================

Javelin is inspired by `DISCUS
<http://tproffen.github.io/DiffuseCode>`_ Until javelin has become
feature equivalent to DISCUS for disordered strcuture creation DISCUS
can still be used to create structures. While javelin can be used to
calculate the diffuse scattering and compare to experimental data.

Javelin can read in discus structure files simply by:

>>> from javelin.io import read_stru
>>> structure = read_stru("tests/data/pzn2.stru")
Found a = 4.06, b = 4.06, c = 4.06, alpha = 90.0, beta = 90.0, gamma = 90.0
Read in these atoms:
Nb     80
O     375
Pb    125
Zn     45
Name: symbol, dtype: int64
>>> print(structure)
Structure(Nb80O375Pb125Zn45, a=4.06, b=4.06, c=4.06, alpha=90.0, beta=90.0, gamma=90.0)

From here it's easy to calculate the scattering.

.. plot::
   :include-source:

   >>> from javelin.io import read_stru  # doctest: +SKIP
   >>> structure = read_stru("../../tests/data/pzn2.stru")  # doctest: +SKIP
   Found a = 4.06, b = 4.06, c = 4.06, alpha = 90.0, beta = 90.0, gamma = 90.0
   Read in these atoms:
   Nb     80
   O     375
   Pb    125
   Zn     45
   Name: symbol, dtype: int64
   >>> from javelin.fourier import Fourier  # doctest: +SKIP
   >>> fourier = Fourier()  # doctest: +SKIP
   >>> fourier.grid.r1 = -2, 2  # doctest: +SKIP
   >>> fourier.grid.r2 = -2, 2  # doctest: +SKIP
   >>> fourier.grid.bins = 201, 201  # doctest: +SKIP
   >>> print(fourier)  # doctest: +SKIP
   Structure         : Structure(Nb80O375Pb125Zn45, a=4.06, b=4.06, c=4.06, alpha=90.0, beta=90.0, gamma=90.0)
   Radiation         : neutron
   Fourier volume    : complete crystal
   Aver. subtraction : False
   <BLANKLINE>
   Reciprocal layer  :
   lower left  corner :     [-2. -2.  0.]
   lower right corner :     [ 2. -2.  0.]
   upper left  corner :     [-2.  2.  0.]
   top   left  corner :     [-2. -2.  1.]
   <BLANKLINE>
   hor. increment     :     [ 0.02  0.    0.  ]
   vert. increment    :     [ 0.    0.02  0.  ]
   top   increment    :     [ 0.  0.  1.]
   <BLANKLINE>
   # of points        :     201 x 201 x 1
   >>> results = fourier.calc(structure)  # doctest: +SKIP
   Working on atom number 8 Total atoms: 375
   Working on atom number 30 Total atoms: 45
   Working on atom number 41 Total atoms: 80
   Working on atom number 82 Total atoms: 125
   >>> results.plot(vmax=2e6)  # doctest: +SKIP
   <matplotlib.collections.QuadMesh object at ...>
