import pytest
import javelin.io as io
from numpy.testing import assert_almost_equal, assert_array_equal
import os


def test_read_mantid_MDHisto_ZrO2nxs():
    pytest.importorskip("h5py")
    filename = os.path.join(os.path.dirname(__file__), 'data', 'ZrO2.nxs')
    ZrO2 = io.read_mantid_MDHisto(filename)
    assert ZrO2.values.shape == (200, 200)
    assert_almost_equal(ZrO2.attrs['a'], 5.150207, decimal=5)
    assert_almost_equal(ZrO2.attrs['alpha'], 90.0, decimal=5)


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

    filename = tmpdir.join('test_file.h5')
    test_array = xr.DataArray([1, 2, 3])
    test_array.attrs['a'] = 5
    test_array.attrs['alpha'] = 90

    # Test save
    io.save_xarray_to_HDF5(test_array, str(filename))
    assert len(tmpdir.listdir()) == 1

    # Test load
    test_data = io.load_HDF5_to_xarray(str(filename))
    assert_array_equal(test_data.values, [1, 2, 3])
    assert test_data.attrs['a'] == 5
    assert test_data.attrs['alpha'] == 90
