Size-effect
===========


.. plot::
   :include-source:

   >>> import numpy as np
   >>> from javelin.structure import Structure
   >>> from javelin.energies import LennardJonesEnergy
   >>> from javelin.modifier import SetDisplacementNormalXYZ
   >>> from javelin.fourier import Fourier
   >>> from javelin.mc import MC
   >>> n = 128
   >>> 
   >>> x = np.random.normal(0, 0.01, size=n**2)
   >>> y = np.random.normal(0, 0.01, size=n**2)
   >>> z = np.zeros(n**2)
   >>> 
   >>> structure = Structure(symbols=np.random.choice(['Na','Cl'],n**2), positions=np.vstack((x,y,z)).T, unitcell=5)
   >>> structure.reindex([n,n,1,1])
   >>> 
   >>> nl = structure.get_neighbors()[0,1,-2,-1]
   >>> 
   >>> e1 = LennardJonesEnergy(1, 1.05, atom_type1=11, atom_type2=11)
   >>> e2 = LennardJonesEnergy(1, 1.0,  atom_type1=11, atom_type2=17)
   >>> e3 = LennardJonesEnergy(1, 0.95, atom_type1=17, atom_type2=17)
   >>> 
   >>> mc = MC()
   >>> mc.add_modifier(SetDisplacementNormalXYZ(0, 0, 0.02, 0, 0.02, 0, 0))
   >>> mc.temperature = 0.001
   >>> mc.cycles = 50
   >>> mc.add_target(nl, e1)
   >>> mc.add_target(nl, e2)
   >>> mc.add_target(nl, e3)
   >>> 
   >>> out = mc.run(structure)  # doctest: +SKIP
   >>> 
   >>> f=Fourier()  # doctest: +SKIP
   >>> f.grid.bins = (201, 201, 1)  # doctest: +SKIP
   >>> f.grid.r1 = -3, 3  # doctest: +SKIP
   >>> f.grid.r2 = -3, 3  # doctest: +SKIP
   >>> f.lots = 8, 8, 1  # doctest: +SKIP
   >>> f.number_of_lots = 256  # doctest: +SKIP
   >>> f.average = True  # doctest: +SKIP
   >>> 
   >>> results1=f.calc(out)  # doctest: +SKIP
   >>> 
   >>> out.replace_atom(11,42)  # doctest: +SKIP
   >>> out.replace_atom(17,11)  # doctest: +SKIP
   >>> out.replace_atom(42,17)  # doctest: +SKIP
   >>> results2=f.calc(out)  # doctest: +SKIP
   >>> 
   >>> out.replace_atom(17,11)  # doctest: +SKIP
   >>> results3=f.calc(out)  # doctest: +SKIP
   >>> fig, axs = plt.subplots(3, 1, figsize=(6.4,14.4))  # doctest: +SKIP
   >>> results1.plot(ax=axs[0])  # doctest: +SKIP
   >>> results2.plot(ax=axs[1])  # doctest: +SKIP
   >>> results3.plot(ax=axs[2])  # doctest: +SKIP
