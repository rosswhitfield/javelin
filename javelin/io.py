"""
==
io
==
"""


def read_mantid_MDHisto(filename):
    """Read the saved MDHisto from from Mantid and returns an xarray.DataArray object"""
    import h5py
    import numpy as np
    import xarray as xr
    from javelin.unitcell import UnitCell
    with h5py.File(filename, "r") as f:
        if ('SaveMDVersion' not in f['MDHistoWorkspace'].attrs or
                f['MDHistoWorkspace'].attrs['SaveMDVersion'] < 2):
            raise RuntimeError("Cannot open "+filename+", must be saved by SaveMD Version 2")

        path = 'MDHistoWorkspace/data/'
        if path+'signal' not in f:
            raise RuntimeError("Cannot open "+path+'signal in '+filename)

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


def read_stru(filename, starting_cell=(1, 1, 1)):
    """Read in a .stru file saved from DISCUS into a javelin Structure

    If the line ncell is not present in the file all the atoms will be
    read into a single cell."""

    from javelin.structure import Structure
    import numpy as np

    with open(filename) as f:
        lines = f.readlines()

    a = b = c = alpha = beta = gamma = 0

    reading_atom_list = False

    ncell = None
    symbols = []
    x = []
    y = []
    z = []

    for l in lines:
        line = l.replace(',', ' ').split()
        if not reading_atom_list:  # Wait for 'atoms' line before reading atoms
            if line[0] == 'cell':
                a, b, c, alpha, beta, gamma = [float(word) for word in line[1:7]]
            elif line[0] == 'ncell':
                ncell = [int(word) for word in line[1:5]]
            elif line[0] == 'atoms':
                if a == 0:
                    print("Cell not found")
                    a = b = c = 1
                    alpha = beta = gamma = 90
                reading_atom_list = True
        else:
            symbol, xx, yy, zz = line[:4]
            symbols.append(symbol)
            x.append(xx)
            y.append(yy)
            z.append(zz)

    print("Found a = {}, b = {}, c = {}, alpha = {}, beta = {}, gamma = {}"
          .format(a, b, c, alpha, beta, gamma))

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    z = np.array(z, dtype=np.float64)
    symbols = np.array(symbols)

    if ncell is not None:
        x -= np.tile(np.repeat(np.array(range(ncell[0])),
                               ncell[3]), ncell[1]*ncell[2]) + starting_cell[0]
        y -= np.tile(np.repeat(np.array(range(ncell[1])),
                               ncell[0]*ncell[3]), ncell[2]) + starting_cell[1]
        z -= np.repeat(np.array(range(ncell[2])),
                       ncell[0]*ncell[1]*ncell[3]) + starting_cell[2]

        # reorder atom arrays, discus stru files have x increment fastest
        # and z slowest, javelin is the opposite
        x = x.reshape(ncell).transpose((2, 1, 0, 3)).flatten()
        y = y.reshape(ncell).transpose((2, 1, 0, 3)).flatten()
        z = z.reshape(ncell).transpose((2, 1, 0, 3)).flatten()
        symbols = symbols.reshape(ncell).transpose((2, 1, 0, 3)).flatten()

    xyz = np.array((x, y, z)).T

    structure = Structure(unitcell=(a, b, c, alpha, beta, gamma),
                          symbols=symbols,
                          positions=xyz,
                          ncells=ncell)

    print("Read in these atoms:")
    print(structure.get_atom_count())

    return structure


def read_stru_to_ase(filename):
    """This function read the legacy DISCUS stru file format into a ASE
    Atoms object.

    :param filename: filename of DISCUS stru file
    :type filename: str
    :return: ASE Atoms object
    :rtype: :class:`ase.Atoms`

    """
    from ase import Atoms
    from ase.geometry import cellpar_to_cell

    with open(filename) as f:
        lines = f.readlines()

    a = b = c = alpha = beta = gamma = 0

    reading_atom_list = False

    symbols = []
    positions = []

    for l in lines:
        line = l.replace(',', ' ').split()
        if not reading_atom_list:  # Wait for 'atoms' line before reading atoms
            if line[0] == 'cell':
                a, b, c, alpha, beta, gamma = [float(x) for x in line[1:7]]
                cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
            if line[0] == 'atoms':
                if a == 0:
                    print("Cell not found")
                    cell = [1, 1, 1]
                reading_atom_list = True
        else:
            symbol, x, y, z = line[:4]
            symbol = symbol.capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])

    # Return ASE Atoms object
    return Atoms(symbols=symbols, scaled_positions=positions, cell=cell)


def numpy_to_vti(array, origin, spacing, filename):
    """This function write a VtkImageData vti file from a numpy array.

    :param array: input array
    :type array: :class:`numpy.ndarray`
    :param origin: the origin of the array
    :type origin: array like object of values
    :param spacing: the step in each dimension
    :type spacing: array like object of values
    :param filename: output filename (.vti)
    :type filename: str
    """

    if array.ndim != 3:
        raise ValueError("Only works with 3 dimensional arrays")

    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type

    vtkArray = numpy_to_vtk(num_array=array.flatten('F'), deep=True,
                            array_type=get_vtk_array_type(array.dtype))

    imageData = vtk.vtkImageData()
    imageData.SetOrigin(origin)
    imageData.SetSpacing(spacing)
    imageData.SetDimensions(array.shape)
    imageData.GetPointData().SetScalars(vtkArray)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()
