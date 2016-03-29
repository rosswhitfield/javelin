def read_mantid_MDHisto(filename):
    """Read the saved MDHisto from from Mantid and returns a javelin Data object"""
    import h5py
    import numpy as np
    from javelin.data import Data
    with h5py.File(filename, "r") as f:
        if ('SaveMDVersion' not in f['MDHistoWorkspace'].attrs or
                f['MDHistoWorkspace'].attrs['SaveMDVersion'] < 2):
            print("Cannot open file, must be saved by SaveMD Version 2")
            return

        path = 'MDHistoWorkspace/data/'
        if path+'signal' not in f:
            print("Can't open "+path+'signal')
            return

        data_set = Data()
        signal = f[path+'signal']
        data = np.array(signal)
        data_set.array = data

        if 'axes' not in signal.attrs:
            print("Can't find axes")
            return data_set
        axes = signal.attrs['axes'].decode().split(":")
        axes.reverse()

        dimensions = len(axes)
        data_set.dim = dimensions
        for a in axes:
            axis = f[path+a]
            data_set.add_axis(a, axis.attrs['units'].decode(),
                              np.array(axis))

        if 'MDHistoWorkspace/experiment0/sample/oriented_lattice' in f:
            lattice = f['MDHistoWorkspace/experiment0/sample/oriented_lattice']
            data_set.set_unit_cell(lattice['unit_cell_a'][0],
                                   lattice['unit_cell_b'][0],
                                   lattice['unit_cell_c'][0],
                                   lattice['unit_cell_alpha'][0],
                                   lattice['unit_cell_beta'][0],
                                   lattice['unit_cell_gamma'][0])

    return data_set
