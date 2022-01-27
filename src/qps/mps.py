"""
Implementation of a MPS-based quantum circuit simulator
"""
import random
import numpy as np

from qps.gate import Gate
from .simulator import StrongSimulator, WeakSimulator


class MPS(StrongSimulator, WeakSimulator):
    """
    A quantum circuit simulator based on Matrix Product State data structure.
    """

    def __init__(self, nqbits: int, max_bound=None):
        self.n = nqbits
        self.max_bound = max_bound

        # Initialize the state as a list of tensors
        self.matrices = [np.array([1, 0]).reshape((1, 2, 1)) for _ in range(self.n)]

    def simulate_gate(self, gate: Gate):
        g = gate.full_matrix

        # Retrieve modified qubit's Gamma
        idx = gate.qubits[0]
        gamma = self.matrices[idx]

        if gate.is_controlled():
            # Retrieve control qubit's Gamma
            idx_c = gate.control_qubit
            gamma_c = self.matrices[idx_c]

            # Store leftmost and rightmost tensor axis sizes
            alpha_prec = gamma_c.shape[0]
            alpha_next = gamma.shape[2]

            # Store contracted qubit matrix dimensions
            n = max(gamma_c.size, gamma.size)
            m = min(gamma_c.size, gamma.size)

            # Contract control and modified qubits
            gamma = np.einsum("ijk,klm->ijlm", gamma_c, gamma)

            # Reshape gate and contracted qubits
            g = g.reshape((2, 2, 4))
            gamma = gamma.reshape((alpha_prec, 4, alpha_next))

            # Apply gate to contracted qubits
            gamma = np.einsum("ijk,lmj->ilmk", gamma, g)

            # Reshape tensor into matrix
            gamma = gamma.reshape((n, m))

            # Perform SVD on matrix
            u, s, v = np.linalg.svd(gamma, full_matrices=False)

            # Remove non-zero singular values
            non_zero_s = np.nonzero(s)
            u = u[:, non_zero_s]
            s = s[non_zero_s]
            v = v[non_zero_s, :]

            u *= s

            if self.max_bound is not None:
                # Truncate SVD matrices
                u = u[:, : self.max_bound]
                v = v[: self.max_bound, :]

            # Assign the two parts of the SVD as the new qubits' tensors
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
        bitstring = ""

        for i in range(self.n):
            prob = np.linalg.norm(self.matrices[i][:, 0, :]) ** 2

            rn = random.random()

            bitstring += "0" if rn < prob else "1"

            if i < self.n - 1:
                vec = np.array([1, 0]) if rn < prob else np.array([0, 1])
                mat = np.einsum("ijk,j->ik", self.matrices[i], vec)

                self.matrices[i + 1] = np.einsum("ik,klm->ilm", mat, self.matrices[i + 1])
                self.matrices[i + 1] /= np.linalg.norm(self.matrices[i + 1])

        return bitstring
