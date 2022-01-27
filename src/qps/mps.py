"""
Implementation of a MPS-based quantum circuit simulator
"""
import numpy as np

from qps.gate import Gate

from .simulator import StrongSimulator, WeakSimulator


class MPS(StrongSimulator, WeakSimulator):
    """
    A quantum circuit simulator based on Matrix Product State data structure.
    """

    def __init__(self, nqbits: int):
        self.n = nqbits

        # Initialize the state as a list of tensors
        self.matrices = [np.array([1, 0]).reshape((1, 2, 1)) for _ in range(self.n)]

    def simulate_gate(self, gate: Gate):
        idx = gate.qubits[0]
        g = gate.full_matrix
        gamma = self.matrices[idx]

        if gate.is_controlled():
            idx_c = gate.control_qubit
            gamma_c = self.matrices[idx_c]

            alpha_prec = gamma_c.shape[0]
            alpha_next = gamma.shape[2]
            n = max(gamma_c.size, gamma.size)
            m = min(gamma_c.size, gamma.size)

            gamma = np.einsum("ijk,klm->ijlm", gamma_c, gamma)

            g = g.reshape((2, 2, 4))
            gamma = gamma.reshape((alpha_prec, 4, alpha_next))

            gamma = np.einsum("ijk,lmj->ilmk", gamma, g)

            gamma = gamma.reshape((n, m))

            u, s, v = np.linalg.svd(gamma, full_matrices=False)
            u *= s

            self.matrices[idx_c] = u.reshape((alpha_prec, 2, -1))
            self.matrices[idx] = v.reshape((-1, 2, alpha_next))

        else:
            self.matrices[idx] = np.einsum("ijk,jl->ilk", gamma, g)

    def get_probability(self, classical_state):
        amplitude = np.ones((1, 1))

        for i, b in enumerate(classical_state):
            vec = np.array([1, 0]) if b == "0" else np.array([0, 1])
            mat = np.einsum("ijk,j->ik", self.matrices[i], vec)

            amplitude = np.einsum("ij,jk->ik", amplitude, mat)

        return abs(amplitude) ** 2

    def get_sample(self):
        return "0" * self.n
