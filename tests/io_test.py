import pytest
import javelin.io as io
from numpy.testing import assert_almost_equal, assert_array_equal
import os


def test_save_read_mantid_MDHisto_ZrO2nxs(tmpdir):
    pytest.importorskip("h5py")

    # Test load
    load_filename = os.path.join(os.path.dirname(__file__), 'data', 'ZrO2.nxs')
    ZrO2 = io.read_mantid_MDHisto(load_filename)
    assert ZrO2.values.shape == (200, 200)
    assert_almost_equal(ZrO2.attrs['unit_cell'].a, 5.150207)
    assert_almost_equal(ZrO2.attrs['unit_cell'].alpha, 1.5707964)

    # Test Save
    save_filename = tmpdir.join('test_file.h5')
    io.save_mantid_MDHisto(ZrO2, str(save_filename))
    assert os.path.isfile(str(save_filename))


def test_save_load_xarray_to_HDF5(tmpdir):
    import xarray as xr

    filename = tmpdir.join('test_file.h5')
    test_array = xr.DataArray([1, 2, 3])

    # Test save
    io.save_xarray_to_HDF5(test_array, str(filename))
    assert len(tmpdir.listdir()) == 1

    # Test load
    test_data = io.load_HDF5_to_xarray(str(filename))
    assert_array_equal(test_data.values, [1, 2, 3])


def test_save_load_xarray_to_HDF5_with_metadata(tmpdir):
    import xarray as xr
    from javelin.unitcell import UnitCell

    filename = tmpdir.join('test_file.h5')
    test_array = xr.DataArray([1, 2, 3])
    test_array.attrs['unit_cell'] = UnitCell(5)

    # Test save
    io.save_xarray_to_HDF5(test_array, str(filename))
    assert len(tmpdir.listdir()) == 1

    # Test load
    test_data = io.load_HDF5_to_xarray(str(filename))
    assert_array_equal(test_data.values, [1, 2, 3])
    assert isinstance(test_data.attrs['unit_cell'], UnitCell)
    assert test_data.attrs['unit_cell'].a == 5


def test_numpy_to_vti(tmpdir):
    pytest.importorskip('vtk')
    import numpy as np
    io.numpy_to_vti(np.ones((10, 10, 10)), (0, 0, 0),
                    (0.1, 0.1, 0.1), str(tmpdir.join('test_file.vti')))
    assert len(tmpdir.listdir()) == 1

    with pytest.raises(ValueError):
        io.numpy_to_vti(np.ones((10, 10)), (0, 0, 0),
                        (0.1, 0.1, 0.1), str(tmpdir.join('test_file.vti')))
