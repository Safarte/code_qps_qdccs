import numpy as np


class Tableau:
    """
    A class representing the stabilizer group of a stabilizer quantum state.
    """

    def __init__(self, dim):
        self.dim = dim
        self.z = np.eye(dim, dtype=np.uint8)
        self.x = np.zeros((dim, dim), dtype=np.uint8)
        self.p = np.zeros(dim, dtype=np.uint8)

    def hadamard(self, qbit):
        """
        Conjugates the group by a H gate
        """
        # SWAP(X[q], Z[q])
        self.z[qbit], self.x[qbit] = self.x[qbit].copy(), self.z[qbit].copy()

        # phase ^= Z[q] & X[q]
        self.p ^= self.z[qbit] & self.x[qbit]

    def phase(self, qbit):
        """
        Conjugates the group by a S gate
        """
        # phase ^= Z[q] & X[q]
        self.p ^= self.z[qbit] & self.x[qbit]

        # Z[q] ^= X[q]
        self.z[qbit] ^= self.x[qbit]

    def cnot(self, control, target):
        """
        Conjugates the group by a CNOT gate
        """
        # phase ^= X[c] & Z[t] & (X[t] ^ Z[c] ^ 1)
        self.p ^= (
            self.x[control] & self.z[target] & (self.x[target] ^ self.z[control] ^ np.ones(self.dim, dtype=np.uint8))
        )

        # Z[c] ^= Z[t]
        # X[t] ^= X[c]
        self.z[control] ^= self.z[target]
        self.x[target] ^= self.x[control]

    def measure(self, qbit):
        """
        Measures a qbit.
        """
        if any(self.x[qbit]):
            # Z_q is not in S(|\psi>)

            # Keep only one column with a X on qubit q
            indices = np.argwhere(self.x[qbit])

            p = indices[0]
            for q in indices[1:]:
                self.mul_col(p[0], q[0])

            self.z[:, p] = np.zeros((self.dim, 1))
            self.z[qbit, p] = 1
            self.x[:, p] = np.zeros((self.dim, 1))

            # 50/50 chance of |0> or |1>
            res = np.random.randint(2)
            self.p[p] = res

            return res

        else:
            # Gaussian elimination to find if Z_q or -Z_q in S(|\psi>)
            b = np.zeros(2 * self.dim, dtype=np.uint8)
            b[qbit] = 1
            T = np.vstack((self.z, self.x))
            for row in range(self.dim):
                indices = np.argwhere(T[:, row])

                if row not in indices:
                    i = indices[0]
                    T[i], T[row] = T[row], T[i]
                    b[i], b[row] = b[row], b[i]
                    del indices[0]
                else:
                    indices = indices[indices != row]

                for i in indices:
                    T[i] ^= T[row]
                    b[i] ^= b[row]

            return sum(self.p & b[: self.dim]) % 2

    def mul_col(self, p, q):
        """Multiplies columns p and q into column q"""
        self.p[q] ^= (self.x[:, p] @ self.z[:, q]) % 2

        self.z[:, q] ^= self.z[:, p]
        self.x[:, q] ^= self.x[:, p]

    def get_circuit(self):
        """
        Generate a circuit that prepares the stabilizer state.
        """
        # TODO :)
