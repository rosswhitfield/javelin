"""
=====
utils
=====
"""


def get_atomic_number_symbol(Z=None, symbol=None):
    """This function returns a tuple of matching arrays of atomic numbers
    (Z) and chemical symbols (symbol).

    :param Z: atomic numbers
    :type Z: int, array like object of int's
    :param symbol: chemical symbols
    :type symbol: str, array like object of str
    :return: arrays of atomic numbers and chemical symbols
    :rtype: tuple of :class:`numpy.ndarray`

    Note: If both Z and symbol are provided the symbol will win out and
    change the Z to match.

    :examples:

    >>> Z, symbol = get_atomic_number_symbol(Z=[12, 24, 26, 48])
    >>> print(Z)
    [12 24 26 48]
    >>> print(symbol)  # doctest: +ALLOW_UNICODE
    ['Mg' 'Cr' 'Fe' 'Cd']

    >>> Z, symbol = get_atomic_number_symbol(symbol=['C', 'H', 'N', 'O'])
    >>> print(Z)
    [6 1 7 8]
    >>> print(symbol)
    ['C' 'H' 'N' 'O']
    """
    import numpy as np
    from periodictable import elements

    if isinstance(Z, int):
        Z = [Z]

    if isinstance(symbol, str):
        symbol = [symbol]

    if symbol is None or len(symbol) == 0:
        if Z is None or len(Z) == 0:
            raise ValueError("Need to provide list of either Z's or symbols.")
        else:
            Z = np.asarray(Z)
            length = len(Z)
            symbol = np.empty(length, dtype='<U2')
            for i in range(length):
                symbol[i] = elements[Z[i]].symbol
            symbol[symbol == 'n'] = 'VD'  # use Z=0 as VOID type
    else:
        symbol = np.asarray(symbol, dtype='<U2')
        length = len(symbol)
        Z = np.empty(length, dtype=np.int64)
        for i in range(length):
            symbol[i] = symbol[i].capitalize()
            if symbol[i] == 'Vd':  # Special case for VOID
                symbol[i] = 'VD'
                Z[i] = 0
            else:
                Z[i] = elements.symbol(symbol[i]).number
    return (Z, symbol)


def get_unitcell(structure):
    """Wrapper to get the unit cell from different structure classes"""
    from javelin.unitcell import UnitCell
    try:  # javelin structure
        return structure.unitcell
    except AttributeError:
        try:  # diffpy structure
            return UnitCell(structure.lattice.abcABG())
        except AttributeError:
            try:  # ASE structure
                from ase.geometry import cell_to_cellpar
                return UnitCell(cell_to_cellpar(structure.cell))
            except (ImportError, AttributeError):
                raise ValueError("Unable to get unit cell from structure")


def get_positions(structure):
    """Wrapper to get the positions from different structure classes"""
    try:  # ASE structure
        return structure.get_scaled_positions()
    except AttributeError:
        try:  # diffpy structure
            return structure.xyz
        except AttributeError:
            raise ValueError("Unable to get positions from structure")


def get_atomic_numbers(structure):
    """Wrapper to get the atomic numbers from different structure classes"""
    try:  # ASE structure
        return structure.get_atomic_numbers()
    except AttributeError:
        try:  # diffpy structure
            atomic_numbers, _ = get_atomic_number_symbol(symbol=structure.element)
            return atomic_numbers
        except AttributeError:
            raise ValueError("Unable to get elements from structure")


def is_structure(structure):
    """Check if an object is a structure that javelin can understand.

    ase.atoms with have cell, get_scaled_positions and get_atomic_numbers attributes
    diffpy.structure with have lattice, xyz, and element attributes
    """
    return (((hasattr(structure, 'cell') or hasattr(structure, 'unitcell')) and
             hasattr(structure, 'get_scaled_positions') and
             hasattr(structure, 'get_atomic_numbers'))
            or
            (hasattr(structure, 'lattice') and
             hasattr(structure, 'xyz') and
             hasattr(structure, 'element')))
