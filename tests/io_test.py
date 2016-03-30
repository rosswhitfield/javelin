import pytest
import javelin.io as io
from numpy.testing import assert_array_almost_equal


def test_read_mantid_MDHisto_ZrO2nxs():
    pytest.importorskip("h5py")
    ZrO2 = io.read_mantid_MDHisto('tests/ZrO2.nxs')
    assert ZrO2.array.shape == (200, 200)
    assert_array_almost_equal(ZrO2.get_unit_cell(),
                              (5.150207, 5.150207, 5.150207, 90.0, 90.0, 90.0),
                              decimal=5)
