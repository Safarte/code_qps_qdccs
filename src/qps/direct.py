"""
Implementation of a naive quantum circuit simulator
"""
import numpy as np

from qps.gate import Gate

from .simulator import StrongSimulator, WeakSimulator


class Direct(StrongSimulator, WeakSimulator):
    """
    A naive quantum circuit simulator based on matrix-vector multiplication.
    """

    def __init__(self, nqbits: int):
        self.n = nqbits

        # Initialize with state |0>
        self.state = np.zeros(1 << self.n, dtype=np.complex128)
        self.state[0] = 1

    def simulate_gate(self, gate: Gate):
        k = len(gate.qubits)

        # Reshape state into tensor with n 2-dimensional axes
        state_tensor = self.state.reshape([2] * self.n)

        if gate.is_controlled():
            # Reorder tensor axes to have control and modified qubits at the start
            state_tensor = np.moveaxis(state_tensor, [gate.control_qubit] + gate.qubits, range(k + 1))

            # Reshape state tensor into a (1, 2**k, 2**(n-k-1)) matrix
            state_tensor = state_tensor.reshape((2, 1 << k, 1 << (self.n - k - 1)))

            # Apply gate if control is 1
            state_tensor[1] = gate.matrix() @ state_tensor[1]

            # Return state tensor to its original shape
            state_tensor = state_tensor.reshape([2] * self.n)
            state_tensor = np.moveaxis(state_tensor, range(k + 1), [gate.control_qubit] + gate.qubits)
        else:
            # Reorder tensor axes to have modified qubits at the start
            state_tensor = np.moveaxis(state_tensor, gate.qubits, range(k))

            # Reshape state tensor into a (2**k, 2**(n-k)) matrix
            state_tensor = state_tensor.reshape((1 << k, 1 << (self.n - k)))

            # Apply gate
            state_tensor = gate.matrix() @ state_tensor

            # Return state tensor to its original shape
            state_tensor = state_tensor.reshape([2] * self.n)
            state_tensor = np.moveaxis(state_tensor, range(k), gate.qubits)

        # Update flattened state
        self.state = state_tensor.reshape(1 << self.n)

    def get_probability(self, classical_state):
        # Return probability of state d: |<d|C|0>|²
        return abs(self.state[int(classical_state, 2)]) ** 2

    def get_sample(self):
        # Sample a random possible state with respect to the state probability distribution
        idx = np.random.choice(range(1 << self.n), p=np.square(np.abs(self.state)))
        return format(idx, "b")
