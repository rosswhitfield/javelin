Working with DiffPy
===================

`DiffPy <http://www.diffpy.org>`_ is a free and open source software
project to provide python software for diffraction analysis and the
study of the atomic structure of materials.

The scattering can be calculated from a
:class:`diffpy.Structure.structure.Structure` directly.

.. plot::
   :include-source:

   >>> from diffpy.Structure import Structure, Lattice, Atom  # doctest: +SKIP
   >>> from javelin.fourier import Fourier  # doctest: +SKIP
   >>> stru = Structure([Atom('C',[0,0,0]), Atom('C',[1,0,0]),
   ...                   Atom('C',[0,1,0]), Atom('C',[1,1,0])],
   ...                  lattice=Lattice(1,1,1,90,90,120))  # doctest: +SKIP
   >>> print(stru)  # doctest: +SKIP
   lattice=Lattice(a=1, b=1, c=1, alpha=90, beta=90, gamma=120)
   C    0.000000 0.000000 0.000000 1.0000
   C    1.000000 0.000000 0.000000 1.0000
   C    0.000000 1.000000 0.000000 1.0000
   C    1.000000 1.000000 0.000000 1.0000
   >>> type(stru)  # doctest: +SKIP
   <class 'diffpy.Structure.structure.Structure'>
   >>> four = Fourier()  # doctest: +SKIP
   >>> four.grid.bins = 201, 201  # doctest: +SKIP
   >>> four.grid.r1 = -2, 2  # doctest: +SKIP
   >>> four.grid.r2 = -2, 2  # doctest: +SKIP
   >>> results = four.calc(stru)  # doctest: +SKIP
   Working on atom number 6 Total atoms: 4
   >>> results.plot()  # doctest: +SKIP
   <matplotlib.collections.QuadMesh object at ...>

Convert structure from diffpy to javelin
----------------------------------------

A :class:`diffpy.Structure.structure.Structure` can be converted to
:class:`javelin.structure.Structure` simply by initializing the
javelin structure from the diffpy structure.

>>> from diffpy.Structure import Structure as diffpy_Structure, Lattice, Atom  # doctest: +SKIP
>>> from javelin.structure import Structure  # doctest: +SKIP
>>> stru = diffpy_Structure([Atom('C',[0,0,0]),Atom('C',[1,1,1])], lattice=Lattice(1,1,1,90,90,120))  # doctest: +SKIP
>>> print(stru)  # doctest: +SKIP
lattice=Lattice(a=1, b=1, c=1, alpha=90, beta=90, gamma=120)
C    0.000000 0.000000 0.000000 1.0000
C    1.000000 1.000000 1.000000 1.0000
>>> type(stru)  # doctest: +SKIP
<class 'diffpy.Structure.structure.Structure'>
>>> javelin_stru = Structure(stru)  # doctest: +SKIP
>>> print(javelin_stru)  # doctest: +SKIP
Structure(C2, a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=120.0)
>>> type(javelin_stru)  # doctest: +SKIP
<class 'javelin.structure.Structure'>

Convert structure from javelin to diffpy
----------------------------------------

>>> type(javelin_stru)  # doctest: +SKIP
<class 'javelin.structure.Structure'>
>>> diffpy_stru = diffpy_Structure([Atom(e, x) for e, x in zip(javelin_stru.element, javelin_stru.xyz)],
...                                lattice=Lattice(*javelin_stru.unitcell.cell))  # doctest: +SKIP
>>> print(diffpy_stru)  # doctest: +SKIP
lattice=Lattice(a=1, b=1, c=1, alpha=90, beta=90, gamma=120)
C    0.000000 0.000000 0.000000 1.0000
C    1.000000 1.000000 1.000000 1.0000
>>> type(diffpy_stru)  # doctest: +SKIP
<class 'diffpy.Structure.structure.Structure'>

File IO
------------

DiffPy file loaders can be utilized by javelin.

>>> from diffpy.Structure.Parsers import getParser  # doctest: +SKIP
>>> from javelin.structure import Structure  # doctest: +SKIP
>>> p = getParser('auto')  # doctest: +SKIP
>>> graphite = Structure(p.parseFile('tests/data/graphite.cif'))  # doctest: +SKIP
>>> print(graphite)  # doctest: +SKIP
Structure(C4, a=2.456, b=2.456, c=6.696, alpha=90.0, beta=90.0, gamma=120.0)
>>> type(graphite)  # doctest: +SKIP
<class 'javelin.structure.Structure'>
>>> pzn = Structure(p.parseFile('tests/data/pzn.stru'))  # doctest: +SKIP
>>> print(pzn)  # doctest: +SKIP
Structure(O9Pb3Nb2Zn1, a=12.18, b=4.06, c=4.06, alpha=90.0, beta=90.0, gamma=90.0)
>>> type(pzn)  # doctest: +SKIP
<class 'javelin.structure.Structure'>
