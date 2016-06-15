def read_mantid_MDHisto(filename):
    """Read the saved MDHisto from from Mantid and returns an xarray.DataArray object"""
    import h5py
    import numpy as np
    import xarray as xr
    from javelin.unitcell import UnitCell
    with h5py.File(filename, "r") as f:
        if ('SaveMDVersion' not in f['MDHistoWorkspace'].attrs or
                f['MDHistoWorkspace'].attrs['SaveMDVersion'] < 2):
            print("Cannot open file, must be saved by SaveMD Version 2")
            return

        path = 'MDHistoWorkspace/data/'
        if path+'signal' not in f:
            print("Can't open "+path+'signal')
            return

        signal = f[path+'signal']
        data = np.array(signal)

        if 'axes' not in signal.attrs:
            print("Can't find axes")
            return xr.DataArray(data)
        axes = signal.attrs['axes'].decode().split(":")

        dims_list = []
        coords_list = []
        for a in axes:
            dims_list.append(a)
            axis = np.array(f[path+a])
            axis = ((axis + np.roll(axis, -1))[:-1])/2  # Hack: Need bin centers
            coords_list.append(np.array(axis))

        data_set = xr.DataArray(data,
                                dims=dims_list,
                                coords=coords_list)

        # Get lattice constants
        oriented_lattice = 'MDHistoWorkspace/experiment0/sample/oriented_lattice'
        if oriented_lattice in f:
            lattice = f[oriented_lattice]
            data_set.attrs['unit_cell'] = UnitCell(lattice['unit_cell_a'][0],
                                                   lattice['unit_cell_b'][0],
                                                   lattice['unit_cell_c'][0],
                                                   lattice['unit_cell_alpha'][0],
                                                   lattice['unit_cell_beta'][0],
                                                   lattice['unit_cell_gamma'][0])

        # Get projection matrix
        W_MATRIX = 'MDHistoWorkspace/experiment0/logs/W_MATRIX/value'
        if W_MATRIX in f:
            data_set.attrs['projection_matrix'] = np.array(f[W_MATRIX]).reshape(3, 3)

    return data_set


def save_mantid_MDHisto(dataArray, filename):
    """Save a file that can be read in using Mantid's LoadMD"""
    import h5py
    import numpy as np
    f = h5py.File(filename, 'w')
    top = f.create_group('MDHistoWorkspace')
    top.attrs['NX_class'] = 'NXentry'
    top.attrs['SaveMDVersion'] = 2

    # Write signal data and axes
    data = top.create_group('data')
    data.attrs['NX_class'] = 'NXdata'
    signal = data.create_dataset('signal', data=dataArray.values, compression="gzip")
    signal.attrs['axes'] = ':'.join(dataArray.coords.dims)
    signal.attrs['signal'] = 1
    for axis in dataArray.coords.dims:
        axis_array = dataArray.coords[axis].values

        # Hack to change to bin boundaries instead of centres.
        bin_size = (axis_array[-1] - axis_array[0])/(len(axis_array) - 1)
        axis_array -= bin_size*0.5
        axis_array = np.append(axis_array, axis_array[-1] + bin_size)

        a = data.create_dataset(axis, data=axis_array)
        a.attrs['frame'] = 'HKL'
        a.attrs['long_name'] = axis
        a.attrs['units'] = 'A^-1'

    data.create_dataset('errors_squared', data=dataArray.values, compression="gzip")
    data.create_dataset('mask', data=np.zeros(dataArray.shape, dtype=np.int8), compression="gzip")
    data.create_dataset('num_events', data=np.ones(dataArray.shape), compression="gzip")
    f.close()


def save_xarray_to_HDF5(dataArray, filename, complib=None):
    """Save the xarray DataArray to HDF file using pandas HDFStore

    attrs will be saved as metadata via pickle

    requries pytables

    complib : {'zlib', 'bzip2', 'lzo', 'blosc', None}, default None"""
    from pandas import HDFStore
    f = HDFStore(filename, mode='w', complib=complib)
    f.put('data', dataArray.to_pandas())
    if len(dataArray.attrs) > 0:
        f.get_storer('data').attrs.metadata = dataArray.attrs
    f.close()


def load_HDF5_to_xarray(filename):
    """Load HDF file into an xarray DataArray using pandas HDFStore

    requries pytables"""
    from pandas import HDFStore
    from xarray import DataArray
    with HDFStore(filename) as f:
        data = f['data']
        if 'metadata' in f.get_storer('data').attrs:
            metadata = f.get_storer('data').attrs.metadata
        else:
            metadata = None
    return DataArray(data, attrs=metadata)


def read_stru(filename):
    import pandas as pd
    import periodictable
    from javelin.structure import Structure
    from javelin.unitcell import UnitCell

    with open(filename) as f:
        lines = f.readlines()

    a = b = c = alpha = beta = gamma = 0
    i_no = 0
    j_no = 0
    k_no = 0
    atom_sites = 0

    reading_atom_list = False

    i = []
    j = []
    k = []
    atom_site = []
    Z = []
    symbols = []
    rel_x = []
    rel_y = []
    rel_z = []
    x = []
    y = []
    z = []

    for l in lines:
        line = l.replace(',', ' ').split()
        if not reading_atom_list:  # Wait for 'atoms' line before reading atoms
            if line[0] == 'cell':
                a, b, c, alpha, beta, gamma = [float(x) for x in line[1:7]]
            elif line[0] == 'ncell':
                i_no, j_no, k_no, atom_sites = [int(x) for x in line[1:5]]
            elif line[0] == 'atoms':
                if a == 0:
                    print("Cell not found")
                    a = b = c = 1
                    alpha = beta = gamma = 90
                reading_atom_list = True
                index_gen = gen_index(i_no, j_no, k_no, atom_sites)
        else:
            ii, jj, kk, ss = next(index_gen)
            i.append(ii)
            j.append(jj)
            k.append(kk)
            atom_site.append(ss)
            symbol, xx, yy, zz = line[:4]
            xx = float(xx)
            yy = float(yy)
            zz = float(zz)
            symbol = symbol.capitalize()
            symbols.append(symbol)
            atomic_number = periodictable.elements.symbol(symbol).number
            Z.append(atomic_number)
            x.append(float(xx))
            y.append(float(yy))
            z.append(float(zz))
            rel_x.append(xx - ii)
            rel_y.append(yy - jj)
            rel_z.append(zz - kk)

    print("Found a = {}, b = {}, c = {}, alpha = {}, beta = {}, gamma = {}"
          .format(a, b, c, alpha, beta, gamma))
    df = pd.DataFrame()
    df['i'] = i
    df['j'] = j
    df['k'] = k
    df['site'] = atom_site
    df['Z'] = Z
    df['symbol'] = symbols
    df['rel_x'] = rel_x
    df['rel_y'] = rel_y
    df['rel_z'] = rel_z
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df = df.set_index(['i', 'j', 'k', 'site'])
    print("Read in these atoms:")
    print(df.symbol.value_counts())

    structure = Structure()
    structure.atoms = df
    structure.unitcell = UnitCell(a, b, c, alpha, beta, gamma)
    return structure


def gen_index(i_number, j_number, k_number, site_number):
    if site_number == 0:  # Put everythin in one unit cell
        site = 0
        while True:
            yield 0, 0, 0, site
            site += 1
    for k in range(k_number):
        for j in range(j_number):
            for i in range(i_number):
                for site in range(site_number):
                    yield i, j, k, site
