import pytest
from javelin.fourier import Fourier
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_Fourier_init():
    four = Fourier()
    assert four.radiation == 'neutrons'
    assert four.structure is None
    assert four.grid.bins == (101, 101)
    assert_array_equal(four.grid.ll, [0.0, 0.0, 0.0])
    assert_array_equal(four.grid.lr, [1.0, 0.0, 0.0])
    assert_array_equal(four.grid.ul, [0.0, 1.0, 0.0])
    assert_array_equal(four.grid.tl, [0.0, 0.0, 1.0])


def test_Fourier_ASE_single_atom():
    pytest.importorskip("ase")
    from ase import Atoms
    atom = Atoms('C2', positions=[(0, 0, 0), (1, 0, 0)])
    four = Fourier()
    four.grid.bins = [21, 2]
    four.grid.lr = [2.0, 0.0, 0.0]
    four.structure = atom
    results = four.calculate()
    expected_result = [1.76804890e+02,   1.59921526e+02,   1.15720303e+02,
                       6.10845872e+01,   1.68833647e+01,   6.62912159e-31,
                       1.68833647e+01,   6.10845872e+01,   1.15720303e+02,
                       1.59921526e+02,   1.76804890e+02,   1.59921526e+02,
                       1.15720303e+02,   6.10845872e+01,   1.68833647e+01,
                       5.96620943e-30,   1.68833647e+01,   6.10845872e+01,
                       1.15720303e+02,   1.59921526e+02,   1.76804890e+02]
    assert_array_almost_equal(results[:, 0], expected_result)
    results = four.calculate_fast()
    assert_array_almost_equal(results[:, 0], expected_result)


def test_Foutier_ASE_C_Ring():
    pytest.importorskip("ase")
    from ase.structure import nanotube
    cnt = nanotube(3, 3, length=1, bond=1.4)
    four = Fourier()
    four.grid.bins = [6, 6, 2]
    four.structure = cnt
    four.grid.ll = [0.0, 0.0, 0.0]
    four.grid.lr = [2.0, 0.0, 0.0]
    four.grid.ul = [0.0, 2.0, 0.0]
    four.grid.tl = [0.0, 0.0, 0.5]
    results = four.calculate()
    expected_result = [[[6.36497605e+03,   3.18248802e+03],
                        [5.19826149e+03,   2.62430530e+03],
                        [2.62792897e+03,   1.44984089e+03],
                        [5.38964665e+02,   4.77287736e+02],
                        [1.77345692e+01,   5.96777257e+01],
                        [5.61330972e+02,   2.13240090e-01]],
                       [[5.19824634e+03,   2.58464390e+03],
                        [4.19804025e+03,   2.08045239e+03],
                        [2.02566353e+03,   1.13472923e+03],
                        [3.43108613e+02,   3.82910378e+02],
                        [4.89415344e+01,   6.29938344e+01],
                        [5.92919277e+02,   4.58410816e+00]],
                       [[2.62726847e+03,   1.23823945e+03],
                        [2.02670889e+03,   8.48260672e+02],
                        [7.91958690e+02,   3.42380372e+02],
                        [3.33463046e+01,   5.27598268e+01],
                        [2.30729880e+02,   5.28798893e-01],
                        [7.98830716e+02,   8.58598324e+00]],
                       [[5.35796499e+02,   1.75090475e+02],
                        [3.44314616e+02,   3.58530973e+01],
                        [3.54586483e+01,   1.01212665e+01],
                        [1.11489239e+02,   1.22750852e+02],
                        [6.43708514e+02,   2.21345300e+02],
                        [1.09996107e+03,   1.70969809e+02]],
                       [[2.07782589e+01,   8.83691575e+01],
                        [4.95216235e+01,   3.06779369e+02],
                        [2.12904292e+02,   6.20110076e+02],
                        [6.00173935e+02,   8.55468985e+02],
                        [1.07400507e+03,   8.34634162e+02],
                        [1.26315713e+03,   5.49870658e+02]],
                       [[6.17667297e+02,   7.27129729e+02],
                        [6.16759220e+02,   1.22358912e+03],
                        [7.37019649e+02,   1.60499636e+03],
                        [9.37950901e+02,   1.66971846e+03],
                        [1.09292588e+03,   1.35970016e+03],
                        [1.03004750e+03,   8.27690724e+02]]]
    assert_array_almost_equal(results, expected_result, 5)
    results = four.calculate_fast()
    assert_array_almost_equal(results, expected_result, 5)
