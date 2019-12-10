Ising Model examples
====================

Creating chemical short-range order

Negative correlation
--------------------

.. plot::
   :include-source:

   >>> import numpy as np
   >>> import xarray as xr
   >>> from javelin.structure import Structure
   >>> from javelin.energies import IsingEnergy
   >>> from javelin.modifier import SwapOccupancy
   >>> from javelin.fourier import Fourier
   >>> from javelin.mc import MC
   >>> n=128
   >>> structure = Structure(symbols=np.random.choice(['Na','Cl'],n**2), positions=[(0., 0., 0.)]*n**2, unitcell=5)
   >>> structure.reindex([n,n,1,1])
   >>> e1 = IsingEnergy(11,17,J=0.5)
   >>> nl = structure.get_neighbors()[0,-1]
   >>> mc = MC()
   >>> mc.add_modifier(SwapOccupancy(0))
   >>> mc.temperature = 1
   >>> mc.cycles = 50
   >>> mc.add_target(nl, e1)
   >>> out = mc.run(structure)  # doctest: +SKIP
   >>> f=Fourier()
   >>> f.grid.bins = 200, 200, 1
   >>> f.grid.r1 = -3,3
   >>> f.grid.r2 = -3,3
   >>> f.lots = 8,8,1
   >>> f.number_of_lots = 256
   >>> f.average = True
   >>> results=f.calc(out)  # doctest: +SKIP
   >>> # plot
   >>> fig, axs = plt.subplots(2, 1, figsize=(6.4,9.6))  # doctest: +SKIP
   >>> xr.DataArray.from_series(out.atoms.Z).plot(ax=axs[0])  # doctest: +SKIP
   >>> results.plot(ax=axs[1])  # doctest: +SKIP
   
Positive correlation
--------------------

.. plot::
   :include-source:

   >>> import numpy as np
   >>> import xarray as xr
   >>> from javelin.structure import Structure
   >>> from javelin.energies import IsingEnergy
   >>> from javelin.modifier import SwapOccupancy
   >>> from javelin.fourier import Fourier
   >>> from javelin.mc import MC
   >>> 
   >>> n=100
   >>> structure = Structure(symbols=np.random.choice(['Na','Cl'],n**2), positions=[(0., 0., 0.)]*n**2, unitcell=5)
   >>> structure.reindex([n,n,1,1])
   >>> e1 = IsingEnergy(11,17,J=-0.5)
   >>> nl = structure.get_neighbors()[0,-1]
   >>> mc = MC()
   >>> mc.add_modifier(SwapOccupancy(0))
   >>> mc.temperature = 1
   >>> mc.cycles = 50
   >>> mc.add_target(nl, e1)
   >>> out = mc.run(structure)  # doctest: +SKIP
   >>> f=Fourier()
   >>> f.grid.bins = 121,121,1
   >>> f.grid.r1 = -3,3
   >>> f.grid.r2 = -3,3
   >>> f.average = True
   >>> results=f.calc(out)  # doctest: +SKIP
   >>> # plot
   >>> fig, axs = plt.subplots(2, 1, figsize=(6.4,9.6))  # doctest: +SKIP
   >>> xr.DataArray.from_series(out.atoms.Z).plot(ax=axs[0])  # doctest: +SKIP
   >>> results.plot(ax=axs[1])  # doctest: +SKIP
   

Getting a desired correlation
-----------------------------

.. plot::
   :include-source:

   >>> import numpy as np
   >>> import xarray as xr
   >>> from javelin.structure import Structure
   >>> from javelin.energies import IsingEnergy
   >>> from javelin.modifier import SwapOccupancy
   >>> from javelin.fourier import Fourier
   >>> from javelin.mc import MC
   >>> 
   >>> n=100
   >>> structure = Structure(symbols=np.random.choice(['Na','Cl'],n**2), positions=[(0., 0., 0.)]*n**2, unitcell=5)
   >>> structure.reindex([n,n,1,1])
   >>> e1 = IsingEnergy(11,17,desired_correlation=0.5)
   >>> nl1 = structure.get_neighbors()[0,-1]
   >>> e2 = IsingEnergy(11,17,desired_correlation=0)
   >>> nl2 = structure.get_neighbors(minD=2.99,maxD=3.01)[0,-1]
   >>> e3 = IsingEnergy(11,17,desired_correlation=-0.5)
   >>> nl3 = structure.get_neighbors()[1,-2]
   >>> mc = MC()
   >>> mc.add_modifier(SwapOccupancy(0))
   >>> mc.temperature = 1
   >>> mc.cycles = 50
   >>> mc.add_target(nl1, e1)
   >>> mc.add_target(nl2, e2)
   >>> mc.add_target(nl3, e3)
   >>> out = mc.run(structure)  # doctest: +SKIP
   >>> f=Fourier()
   >>> f.grid.bins = 121,121,1
   >>> f.grid.r1 = -3,3
   >>> f.grid.r2 = -3,3
   >>> f.average = True
   >>> results=f.calc(out)  # doctest: +SKIP
   >>> # plot
   >>> fig, axs = plt.subplots(2, 1, figsize=(6.4,9.6))  # doctest: +SKIP
   >>> xr.DataArray.from_series(out.atoms.Z).plot(ax=axs[0])  # doctest: +SKIP
   >>> results.plot(ax=axs[1])  # doctest: +SKIP
