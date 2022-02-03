"""
Implementation of a MPS-based quantum circuit simulator
"""
import random
import numpy as np

from qps.gate import Gate
from .simulator import StrongSimulator, WeakSimulator

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def normalize(array):
    return array / np.linalg.norm(array)


class MPS(StrongSimulator, WeakSimulator):
    """
    A quantum circuit simulator based on Matrix Product State data structure.
    """

    def __init__(self, nqbits: int, max_bound=None):
        self.n = nqbits
        self.max_bound = max_bound

        # Initialize the state as a list of tensors
        self.matrices = [np.array([1, 0]).reshape((1, 2, 1)) for _ in range(self.n)]

    def apply_one_qubit_gate(self, target, gate):
        gamma = self.matrices[target]

        self.matrices[target] = np.einsum("ijk,jl->ilk", gamma, gate)

    def apply_two_qubits_gate(self, control, target, gate):
        # Retrieve qubits Gamma
        gamma = self.matrices[target]
        gamma_c = self.matrices[control]

        # Store leftmost and rightmost tensor axis sizes
        alpha_prec = gamma_c.shape[0]
        alpha_next = gamma.shape[2]

        # Store contracted qubit matrix dimensions
        n = max(alpha_prec, alpha_next)
        m = min(alpha_prec, alpha_next)

        # Contract control and modified qubits
        gamma = np.einsum("ijk,klm->ijlm", gamma_c, gamma)

        # Reshape gate and contracted qubits
        g = gate.reshape((2, 2, 4))
        gamma = gamma.reshape((alpha_prec, 4, alpha_next))

        # Apply gate to contracted qubits
        gamma = np.einsum("ijk,lmj->ilmk", gamma, g)

        # Reshape tensor into matrix
        gamma = gamma.reshape((2 * n, 2 * m))

        # Perform SVD on matrix
        u, s, v = np.linalg.svd(gamma, full_matrices=False)

        indices = s.argsort()[::-1]
        if self.max_bound is not None:
            # Truncate SVD matrices
            indices = indices[: self.max_bound]

        # Remove non-zero singular values
        non_zero_s = np.nonzero(s[indices])
        u = u[:, non_zero_s]
        s = s[non_zero_s]
        v = v[non_zero_s, :]

        u *= s

        if self.max_bound is not None:
            # Truncate SVD matrices
            u = u[:, : self.max_bound]
            v = v[: self.max_bound, :]

        # Assign the two parts of the SVD as the new qubits' tensors
        if self.matrices[control].size >= self.matrices[target].size:
            self.matrices[control] = u.reshape((alpha_prec, 2, -1))
            self.matrices[target] = v.reshape((-1, 2, alpha_next))
        else:
            self.matrices[control] = v.reshape((alpha_prec, 2, -1))
            self.matrices[target] = u.reshape((-1, 2, alpha_next))

    def simulate_gate(self, gate: Gate):
        control = gate.control_qubit
        target = gate.qubits[0]

        if gate.is_controlled():
            if target - control > 1:
                # Apply SWAP cascades
                for i in range(control, target - 1):
                    self.apply_two_qubits_gate(i, i + 1, SWAP)

                self.apply_two_qubits_gate(target - 1, target, gate.full_matrix)

                for i in range(target - 2, control - 1, -1):
                    self.apply_two_qubits_gate(i, i + 1, SWAP)
            else:
                self.apply_two_qubits_gate(control, target, gate.full_matrix)
        else:
            self.apply_one_qubit_gate(target, gate.full_matrix)

    def get_probability(self, classical_state):
        amplitude = np.ones((1, 1))

        for i, b in enumerate(classical_state):
            # Contract |0> or |1> into qubit's tensor
            vec = np.array([1, 0]) if b == "0" else np.array([0, 1])
            mat = np.einsum("ijk,j->ik", self.matrices[i], vec)

            # Contract qubit's tensor into amplitude
            amplitude = np.einsum("ij,jk->ik", amplitude, mat)

        return abs(amplitude) ** 2

    def get_sample(self):
        bitstring = ""
        contracted = np.array([1])

        for i in range(self.n):
            contracted = np.einsum("i,ikl->kl", contracted, self.matrices[i])

            prob = np.linalg.norm(contracted[0, :]) ** 2

            rn = random.random()

            bitstring += "0" if rn < prob else "1"
            contracted = contracted[0, :] if rn < prob else contracted[0, :]
            contracted = normalize(contracted)

        return bitstring
