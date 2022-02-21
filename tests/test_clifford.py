import pytest
from qps.clifford import Tableau


def test_hh():
    tab = Tableau(1)
    tab.hadamard(0)
    tab.hadamard(0)
    assert tab.measure(0) == 0


def test_x():
    tab = Tableau(1)
    tab.hadamard(0)
    tab.phase(0)
    tab.phase(0)
    tab.hadamard(0)
    assert tab.measure(0) == 1


def test_measure():
    for _ in range(100):
        tab = Tableau(1)
        tab.hadamard(0)
        assert tab.measure(0) == tab.measure(0)


@pytest.mark.parametrize("nbits", [*range(10, 101, 10)])
def test_ghz(nbits):
    tab = Tableau(nbits)
    tab.hadamard(0)
    for i in range(nbits - 1):
        tab.cnot(i, i + 1)
    res = [tab.measure(i) for i in range(nbits)]
    assert all(res) or not any(res)


def test_teleport():
    tab = Tableau(3)

    # Alice qubit to 1
    tab.hadamard(0)
    tab.phase(0)
    tab.phase(0)
    tab.hadamard(0)

    # Teleportation
    tab.hadamard(1)
    tab.cnot(1, 2)
    tab.cnot(0, 1)
    tab.hadamard(0)

    if tab.measure(1):
        # X
        tab.hadamard(2)
        tab.phase(2)
        tab.phase(2)
        tab.hadamard(2)

    if tab.measure(0):
        # Z
        tab.phase(2)
        tab.phase(2)

    assert tab.measure(2) == 1
