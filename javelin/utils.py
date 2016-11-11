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

    """
    import numpy as np
    from periodictable import elements

    if isinstance(Z, int):
        Z = [Z]

    if isinstance(symbol, str):
        symbol = [symbol]

    if np.count_nonzero(symbol) == 0:
        if np.count_nonzero(Z) == 0:
            raise ValueError("Need to provide list of either Z's or symbols.")
        else:
            Z = np.asarray(Z)
            length = len(Z)
            symbol = np.empty(length, dtype='<U2')
            for i in range(length):
                symbol[i] = elements[Z[i]].symbol
    else:
        symbol = np.asarray(symbol)
        length = len(symbol)
        Z = np.empty(length, dtype=np.int64)
        for i in range(length):
            symbol[i] = symbol[i].capitalize()
            Z[i] = elements.symbol(symbol[i]).number
    return (Z, symbol)
