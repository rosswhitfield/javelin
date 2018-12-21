from javelin.energies import (Energy, IsingEnergy,
                              DisplacementCorrelationEnergy,
                              SpringEnergy, LennardJonesEnergy)
from numpy.testing import assert_almost_equal


def test_Energy():
    e = Energy()
    assert str(e) == "Energy()"
    assert e.evaluate(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == 0


def test_IsingEnergy():
    e = IsingEnergy(13, 42, -0.5)
    assert str(e) == "IsingEnergy(Atom1=13,Atom2=42,J=-0.5)"
    assert e.atom1 == 13
    assert e.atom2 == 42
    assert e.J == -0.5
    assert e.evaluate(1, 0, 0, 0,
                      1, 0, 0, 0,
                      0, 0, 0) == 0
    assert e.evaluate(13, 0, 0, 0,
                      42, 0, 0, 0,
                      0, 0, 0) == 0.5
    assert e.evaluate(42, 0, 0, 0,
                      42, 0, 0, 0,
                      0, 0, 0) == -0.5


def test_DisplacementCorrelationEnergy():
    e = DisplacementCorrelationEnergy(-0.5)
    assert str(e) == "DisplacementCorrelationEnergy()"
    assert e.J == -0.5
    assert e.evaluate(0, 1, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 0) == -0.5
    assert e.evaluate(0, -1, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 0) == 0.5
    assert_almost_equal(e.evaluate(0, -1, 0, 0,
                                   0, 1, 1, 0,
                                   0, 0, 0), 0.35355339)
    assert e.evaluate(0, -1, 1, 0,
                      0, 1, 1, 0,
                      0, 0, 0) == 0
    assert e.evaluate(0, 1, 1, 1,
                      0, 0, 0, 0,
                      0, 0, 0) == 0
    assert_almost_equal(e.evaluate(0, -1, 1, 1,
                                   0, 1, 0, 0.5,
                                   0, 0, 0), 0.12909944)
    assert_almost_equal(e.evaluate(0, -0.1, 0.2, -0.3,
                                   0, 0.4, -0.5, 0.6,
                                   0, 0, 0), 0.48731592)


def test_SpringEnergy():
    e = SpringEnergy(0.5, 1.1)
    assert str(e) == "SpringEnergy(K=0.5,desired=1.1,atoms=all)"
    assert e.K == 0.5
    assert e.desired == 1.1
    assert e.evaluate(0, 0, 0, 0,
                      0, 0.1, 0, 0,
                      1, 0, 0) == 0
    assert_almost_equal(e.evaluate(0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   1, 0, 0), 0.005)
    assert_almost_equal(e.evaluate(0, 0, 0, 0,
                                   0, 0.2, 0, 0,
                                   1, 0, 0), 0.005)
    assert_almost_equal(e.evaluate(0, -0.3, -0.2, -0.1,
                                   0, 0.1, 0.2, 0.3,
                                   3, 2, 1), 5.41501035)

    # with atom types
    e = SpringEnergy(0.5, 1.1, 11, 17)
    assert str(e) == "SpringEnergy(K=0.5,desired=1.1,atoms=11-17)"
    assert_almost_equal(e.evaluate(99, 0, 0, 0,
                                   99, 0, 0, 0,
                                   1, 0, 0), 0)
    assert_almost_equal(e.evaluate(11, 0, 0, 0,
                                   11, 0, 0, 0,
                                   1, 0, 0), 0)
    assert_almost_equal(e.evaluate(11, 0, 0, 0,
                                   17, 0, 0, 0,
                                   1, 0, 0), 0.005)
    assert_almost_equal(e.evaluate(17, 0, 0, 0,
                                   11, 0, 0, 0,
                                   1, 0, 0), 0.005)
    assert_almost_equal(e.evaluate(17, 0, 0, 0,
                                   17, 0, 0, 0,
                                   1, 0, 0), 0)


def test_LennardJonesEnergy():
    e = LennardJonesEnergy(0.5, 1.1)
    assert str(e) == "LennardJonesEnergy(D=0.5,desired=1.1,atoms=all)"
    assert e.D == 0.5
    assert e.desired == 1.1
    assert e.evaluate(0, 0, 0, 0,
                      0, 0.1, 0, 0,
                      1, 0, 0) == -0.5
    assert_almost_equal(e.evaluate(0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   1, 0, 0), -0.2023468)
    assert_almost_equal(e.evaluate(0, 0, 0, 0,
                                   0, 0.1, 0, 0,
                                   1, 0, 0), -0.5)
    assert_almost_equal(e.evaluate(0, 0, 0, 0,
                                   0, 0.2, 0, 0,
                                   1, 0, 0), -0.4172944)
    assert_almost_equal(e.evaluate(0, -0.3, -0.2, -0.1,
                                   0, 0.1, 0.2, 0.3,
                                   3, 2, 1), -0.00024716)

    # With atom types
    e = LennardJonesEnergy(0.5, 1.1, 11, 17)
    assert str(e) == "LennardJonesEnergy(D=0.5,desired=1.1,atoms=11-17)"
    assert_almost_equal(e.evaluate(99, 0, 0, 0,
                                   99, 0.1, 0, 0,
                                   1, 0, 0), 0)
    assert_almost_equal(e.evaluate(11, 0, 0, 0,
                                   11, 0.1, 0, 0,
                                   1, 0, 0), 0)
    assert_almost_equal(e.evaluate(11, 0, 0, 0,
                                   17, 0.1, 0, 0,
                                   1, 0, 0), -0.5)
    assert_almost_equal(e.evaluate(17, 0, 0, 0,
                                   11, 0.1, 0, 0,
                                   1, 0, 0), -0.5)
    assert_almost_equal(e.evaluate(17, 0, 0, 0,
                                   17, 0.1, 0, 0,
                                   1, 0, 0), 0)
