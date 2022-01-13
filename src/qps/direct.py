"""
Implementation of a naive quantum circuit simulator
"""
import numpy as np

from .simulator import StrongSimulator, WeakSimulator


class Direct(StrongSimulator, WeakSimulator):
    """
    A naive quantum circuit simulator based on matrix-vector multiplication.
    """

    def __init__(self, nqbits):
        self.n = nqbits

        # Initialize with state |0>
        self.state = np.zeros(2 ** self.n)
        self.state[0] = 1

    def simulate_gate(self, gate, qubits):
        k = len(qubits)

        # Reshape state into tensor with n 2-dimensional axes
        state_tensor = self.state.reshape([2] * self.n)

        # Reorder tensor axes to have modified qubits at the start
        state_tensor = np.moveaxis(state_tensor, qubits, range(k))

        # Reshape state tensor into matrix
        m = state_tensor.reshape((2 ** k, 2 ** (self.n - k)))

        # Apply gate
        m = gate @ m

        # Return state tensor to original shape
        state_tensor = m.reshape([2] * self.n)
        state_tensor = np.moveaxis(state_tensor, range(k), qubits)

        self.state = state_tensor.reshape(2 ** self.n)

    def get_probability(self, classical_state):
        # Return probability of state |<d|C|0>|²
        return abs(self.state[int(classical_state, 2)]) ** 2

    def get_sample(self):
        # Sample a random possible state with respect to the probability distribution
        idx = np.random.choice(range(2 ** self.n), p=np.square(np.abs(self.state)))
        return format(idx, "b")
