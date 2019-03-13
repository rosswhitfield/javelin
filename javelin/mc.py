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
    P = \exp(-\Delta E / kT)

.. plot::

    x = np.linspace(-1,5,600)
    for kT in [0.1, 0.5, 1, 2, 5]:
        plt.plot(x, np.minimum(1, np.exp(-x/kT)), label="$kT={}$".format(kT))
    plt.xlabel("ΔE")
    plt.ylabel("P")
    plt.legend()
    plt.title("Probabilty change is accepted for given ΔE with different $kT$")
"""

import numpy as np
import copy
from javelin.mccore import mcrun, Target
from javelin.energies import Energy
from javelin.modifier import BaseModifier


class MC:
    """MonteCarlo class

    For the Monte Carlo simulations to run you need to provide an
    input structure, target (neighbor and energy set) and modifier. A
    basic, do nothing, example is shown:

    >>> from javelin.structure import Structure
    >>> from javelin.modifier import BaseModifier
    >>> from javelin.energies import Energy
    >>> structure = Structure(symbols=['Na','Cl'],positions=[[0,0,0],[0.5,0.5,0.5]])
    >>>
    >>> energy = Energy()
    >>> neighbors = structure.get_neighbors()
    >>>
    >>> mc = MC()
    >>> mc.add_modifier(BaseModifier(0))
    >>> mc.temperature = 1
    >>> mc.cycles = 2
    >>> mc.add_target(neighbors, energy)
    >>>
    >>> new_structure = mc.run(structure)
    <BLANKLINE>
    Cycle = 0, temperature = 1.0
    Accepted 0 good, 1 neutral (dE=0) and 0 bad out of 1
    <BLANKLINE>
    Cycle = 1, temperature = 1.0
    Accepted 0 good, 1 neutral (dE=0) and 0 bad out of 1
    >>>

    """

    def __init__(self):
        self.__cycles = 100
        self.__temp = 1
        self.__targets = []
        self.__modifier = []
        self.__iterations = None

    def __str__(self):
        return """Number of cycles = {}
Temperature[s] = {}
Structure modfifiers are {}""".format(self.cycles,
                                      self.temperature,
                                      [str(m) for m in self.modifier])

    @property
    def cycles(self):
        """The number of cycles to perform.
        """
        return self.__cycles

    @cycles.setter
    def cycles(self, cycles):
        try:
            self.__cycles = int(cycles)
        except ValueError:
            raise ValueError('cycles must be an int')

    @property
    def iterations(self):
        """The number of iterations (site modifications) to perform for each
        cycle. Default is equal to the number of unitcells in the
        structure.
        """
        return self.__iterations

    @iterations.setter
    def iterations(self, iterations):
        try:
            self.__iterations = int(iterations)
        except ValueError:
            raise ValueError('iterations must be an int')

    @property
    def temperature(self):
        r"""Temperature parameter (:math:`kT`) which changes the probability
        (P) a energy change of :math:`\Delta E` is accepted

           .. math::
               P = \exp(-\Delta E / kT)

        Temperature can be a single value for all cycles or you can
        provide a different temperature for each cycle. This allows
        you to do quenching of the disorder. If you provide more
        temperatures than cycles then only the first temperatures
        corresponding to the number of cycles are used. If there are
        more cycles than temperature than for remaining cycles the
        last temperature in the list will be used.

        >>> mc = MC()
        >>> mc.temperature = 0.1
        >>> print(mc)
        Number of cycles = 100
        Temperature[s] = 0.1
        Structure modfifiers are []
        >>>
        >>> mc.temperature = np.linspace(1, 0, 6)
        >>> print(mc)
        Number of cycles = 100
        Temperature[s] = [ 1.   0.8  0.6  0.4  0.2  0. ]
        Structure modfifiers are []
        """
        return self.__temp

    @temperature.setter
    def temperature(self, temperature):
        try:
            self.__temp = float(temperature)
        except (ValueError, TypeError):
            try:
                self.__temp = np.asarray(temperature, dtype=float)
            except ValueError:
                raise ValueError('temperature must be real numbers')

    @property
    def modifier(self):
        """This is how the structure is to be modified, must be of type
        :class:`javelin.modifier.BaseModifier`.

        """
        return self.__modifier

    @modifier.setter
    def modifier(self):
        raise ValueError("You must use add_modifier to add a modifier")

    def add_modifier(self, modifier):
        if isinstance(modifier, BaseModifier):
            self.__modifier.append(modifier)
        else:
            raise ValueError("modifier must be an instance of javelin.modifier.BaseModifier")

    def add_target(self, neighbors, energy):
        """This will add an energy calculation and neighbour pair that will be
        used to calculate and energy for each modification step. You
        can add as many targets as you like.

        :param neighbour: neighbour for which the energy will be calculated over
        :type neighbour: :class:`javelin.neighborlist.NeighborList` or `n x 5` array
            of neighbor vectors
        :param energy: the energy function that will be calculated for each neighbor
        :type energy: :class:`javelin.energies.Energy`
        """
        if isinstance(energy, Energy):
            try:
                self.__targets.append(
                    Target(np.asarray(neighbors).astype(np.intp).reshape((-1, 5)),
                           energy))
            except ValueError:
                raise ValueError("neighbors must be javelin.neighborlist.NeighborList or "
                                 "`n x 5` array of neighbor vectors where dtype=int")
        else:
            raise ValueError("energy must be an instance of javelin.energies.Energy")

    def delete_targets(self):
        """This will remove all previously set targets
        """
        self.__targets = []

    def run(self, structure, inplace=False):  # noqa: C901
        """Execute the Monte Carlo routine. You must provide the structure to
        modify as a parameter. This will by default this will return a
        new :class:`javelin.structure.Structure` with the results, to
        modify the provided structure in place set `inplace=True`

        :param structure: structure to run the Monte Carlo on
        :type structure: :class:`javelin.structure.Structure`

        """
        if structure is None:
            raise ValueError("Need to provide input structure")

        if len(self.__targets) == 0:
            raise ValueError("You must add targets to the MC object with add_target")

        if len(self.__modifier) == 0:
            raise ValueError("You must add a modifier to the MC object with add_modifier")

        if not inplace:
            structure = copy.deepcopy(structure)

        shape = (len(structure.atoms.index.levels[0]),
                 len(structure.atoms.index.levels[1]),
                 len(structure.atoms.index.levels[2]),
                 len(structure.atoms.index.levels[3]))

        iterations = self.__iterations or len(structure.atoms.index.droplevel(3).drop_duplicates())

        temps = np.atleast_1d(self.temperature)
        temps = np.pad(temps, (0, max(0, self.cycles-len(temps))), mode='edge')
        for cycle in range(self.cycles):
            temp = temps[cycle]
            print('\nCycle = {}, temperature = {}'.format(cycle, temp))

            # Do feedback
            for target in self.__targets:
                if (target.energy.correlation_type == 0 or
                   np.isnan(target.energy.desired_correlation)):
                    continue
                elif target.energy.correlation_type == 1:
                    correlation = structure.get_occupational_correlation(target.neighbors,
                                                                         target.energy.atom1)
                elif target.energy.correlation_type == 2:
                    correlation = structure.get_displacement_correlation(target.neighbors)
                else:
                    raise RuntimeError("Unknown correlation type for energy {}"
                                       .format(target.energy))
                target.energy.J += (correlation - target.energy.desired_correlation)
                print("Correlations of {} with neighbors:\n{}\nis {:.5} "
                      "for desired correlation of {:.5}. Setting J to {:.5}"
                      .format(target.energy,
                              np.asarray(target.neighbors),
                              correlation,
                              target.energy.desired_correlation,
                              target.energy.J))

            # Do MC loop
            good, neutral, bad = mcrun(np.array(self.__modifier),
                                       np.array(self.__targets),
                                       iterations,
                                       temp,
                                       structure.get_atomic_numbers().reshape(shape),
                                       structure.x.reshape(shape),
                                       structure.y.reshape(shape),
                                       structure.z.reshape(shape))
            print("Accepted {} good, {} neutral (dE=0) and {} bad out of {}"
                  .format(good, neutral, bad, iterations))

        # Update symbols after Z's were changed
        structure.update_atom_symbols()

        if not inplace:
            return structure
