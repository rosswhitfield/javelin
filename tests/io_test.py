import pytest
import javelin.io as io
from numpy.testing import assert_almost_equal
import os


def test_read_mantid_MDHisto_ZrO2nxs():
    pytest.importorskip("h5py")
    filename = os.path.join(os.path.dirname(__file__), 'data', 'ZrO2.nxs')
    ZrO2 = io.read_mantid_MDHisto(filename)
    assert ZrO2.values.shape == (200, 200)
    assert_almost_equal(ZrO2.attrs['a'], 5.150207, decimal=5)
    assert_almost_equal(ZrO2.attrs['alpha'], 90.0, decimal=5)
