"""
Quantum gate interface and some implementations
"""
import abc

import numpy as np


class Gate(abc.ABC):
    def __init__(self, qubits, control_qubit=-1):
        """
        Initialize quantum gate
        """
        self._qubits = qubits
        self._control_qubit = control_qubit

    @property
    @abc.abstractmethod
    def is_controlled(self):
        """
        Is the gate controlled ?
        """

    @property
    def control_qubit(self):
        """
        Index of the control qubit if controlled, else -1
        """
        return self._control_qubit

    @property
    def qubits(self):
        """
        List of the qubits on which the gate is applied
        """
        return self._qubits

    @property
    @abc.abstractmethod
    def matrix(self):
        """
        Matrix representation of the gate
        """

    @property
    def controlled_matrix(self):
        n = 1 << len(self.qubits)
        return np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), self.matrix()]])


class H(Gate):
    def is_controlled(self):
        return False

    def matrix(self):
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2.0)


class CNot(Gate):
    def is_controlled(self):
        return True

    def matrix(self):
        return np.array([[0, 1], [1, 0]])


class X(Gate):
    def is_controlled(self):
        return False

    def matrix(self):
        return np.array([[0, 1], [1, 0]])


class Id(Gate):
    def is_controlled(self):
        return False

    def matrix(self):
        return np.eye(1 << len(self.qubits))


class CZ(Gate):
    def is_controlled(self):
        return True

    def matrix(self):
        return np.array([[1, 0], [0, -1]])


class RX(Gate):
    def __init__(self, qubits, theta, control_qubit=-1):
        self._theta = theta
        super().__init__(qubits, control_qubit)

    def is_controlled(self):
        return False

    def matrix(self):
        return np.cos(self._theta / 2) * np.eye(2) - 1j * np.sin(self._theta / 2) * np.array([[0, 1], [1, 0]])


class RZ(Gate):
    def __init__(self, qubits, theta, control_qubit=-1):
        self._theta = theta
        super().__init__(qubits, control_qubit)

    def is_controlled(self):
        return False

    def matrix(self):
        return np.diag([np.exp(1j * self._theta / 2), np.exp(-1j * self._theta / 2)])
