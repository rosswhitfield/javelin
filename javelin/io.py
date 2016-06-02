def read_mantid_MDHisto(filename):
    """Read the saved MDHisto from from Mantid and returns an xarray.DataArray object"""
    import h5py
    import numpy as np
    import xarray as xr
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

        if 'MDHistoWorkspace/experiment0/sample/oriented_lattice' in f:
            lattice = f['MDHistoWorkspace/experiment0/sample/oriented_lattice']
            data_set.attrs['a'] = lattice['unit_cell_a'][0]
            data_set.attrs['b'] = lattice['unit_cell_b'][0]
            data_set.attrs['c'] = lattice['unit_cell_c'][0]
            data_set.attrs['alpha'] = lattice['unit_cell_alpha'][0]
            data_set.attrs['beta'] = lattice['unit_cell_beta'][0]
            data_set.attrs['gamma'] = lattice['unit_cell_gamma'][0]

    return data_set


def save_xarray_to_HDF5(dataArray, filename, complib=None):
    """Save the xarray DataArray to HDF file using pandas HDFStore

    attrs will be saved as metadata via pickle

    requries pytables

    complib : {'zlib', 'bzip2', 'lzo', 'blosc', None}, default None"""
    from pandas import HDFStore
    file = HDFStore(filename, mode='w', complib=complib)
    file.put('data', dataArray.to_pandas())
    if len(dataArray.attrs) > 0:
        file.get_storer('data').attrs.metadata = dataArray.attrs
    file.close()


def load_HDF5_to_xarray(filename):
    """Load HDF file into an xarray DataArray using pandas HDFStore

    requries pytables"""
    from pandas import HDFStore
    from xarray import DataArray
    with HDFStore(filename) as file:
        data = file['data']
        if 'metadata' in file.get_storer('data').attrs:
            metadata = file.get_storer('data').attrs.metadata
        else:
            metadata = None
    return DataArray(data, attrs=metadata)
