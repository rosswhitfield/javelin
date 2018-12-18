r"""
==
mc
==

.. graphviz::

   digraph mc {
      initial [label="Initial configuration\n\nCalculate Energy E", shape=box];
      change [label="Change a variable at random", shape=box];
      calc [label="Calculate ΔE", shape=box];
      dE [label="ΔE ?", shape=diamond];
      initial -> change -> calc -> dE;

      keep [label="Keep new\nconfiguration", shape=box];
      keep2 [label="Keep new\nconfiguration\nwith probability P", shape=box];
      dE -> keep [label="ΔE < 0"];
      dE -> keep2 [label="ΔE > 0"];

      repeat [label="Repeat until E reaches minimum", shape=box];
      keep -> repeat;
      keep2 -> repeat;
      repeat -> change;
   }

where

.. math::
    P = \frac{\exp(-\Delta E / kT)}{1+\exp(-\Delta E / kT)}
"""

import numpy as np
import copy
from javelin.mccore import mcrun, Target
from javelin.energies import Energy
from javelin.modifier import BaseModifier


class MC:
    """MonteCarlo class

    Requires:

    input structure, target, move generator
    """

    def __init__(self):
        self.__cycles = 100
        self.__temp = 1
        self.__targets = []
        self.__modifier = None
        self.__iterations = 1

    def __str__(self):
        return """Number of cycles = {}
Temperature = {}
Structure modfifier is {}""".format(self.cycles,
                                    self.temperature,
                                    self.modifier)

    @property
    def cycles(self):
        return self.__cycles

    @cycles.setter
    def cycles(self, cycles):
        try:
            self.__cycles = int(cycles)
        except ValueError:
            raise ValueError('cycles must be an int')

    @property
    def temperature(self):
        return self.__temp

    @temperature.setter
    def temperature(self, temperature):
        try:
            self.__temp = float(temperature)
        except ValueError:
            raise ValueError('temperature must be a real number')

    @property
    def modifier(self):
        return self.__modifier

    @modifier.setter
    def modifier(self, modifier):
        if isinstance(modifier, BaseModifier):
            self.__modifier = modifier
        else:
            raise ValueError("modifier must be an instance of javelin.modifier.BaseModifier")

    def add_target(self, neighbours, energy):
        if isinstance(energy, Energy):
            try:
                self.__targets.append(
                    Target(np.asarray(neighbours).astype(np.intp).reshape((-1, 5)),
                           energy))
            except ValueError:
                raise ValueError("neighbours must be javelin.neighborlist.NeighborList or "
                                 "`n x 5` array of neighbor vectors where dtype=int")
        else:
            raise ValueError("energy must be an instance of javelin.energies.Energy")

    def delete_targets(self):
        self.__targets = []

    def run(self, structure, inplace=False):
        if structure is None:
            raise ValueError("Need to provide input structure")

        if not inplace:
            structure = copy.copy(structure)

        shape = (len(structure.atoms.index.levels[0]),
                 len(structure.atoms.index.levels[1]),
                 len(structure.atoms.index.levels[2]),
                 len(structure.atoms.index.levels[3]))

        for cycle in range(self.cycles):
            print('Cycle = {}'.format(cycle))
            accepted = mcrun(self.modifier,
                             np.array(self.__targets),
                             len(structure.atoms)*self.__iterations,
                             self.temperature,
                             structure.get_atomic_numbers().reshape(shape),
                             structure.x.reshape(shape),
                             structure.y.reshape(shape),
                             structure.z.reshape(shape))
            print("Accepted {} out of {}".format(accepted, len(structure.atoms)))

        # Update symbols after Z's were changed
        structure.update_atom_symbols()

        if not inplace:
            return structure
