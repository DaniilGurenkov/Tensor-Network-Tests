"""
Matrix product state (MPS) quantum circuit simulator.

Stores an n-qubit state as a chain of small tensors instead of a single
2^n vector. Two-qubit gates are applied by contracting neighboring tensors
and splitting them back apart with an SVD, dropping small singular values.
"""

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
CNOT = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)


class MPS:
    def __init__(self, n_qubits, max_bond=64, trunc_tol=1e-10):
        # every qubit starts in |0>; tensors have shape (left bond, 2, right bond)
        self.n = n_qubits
        self.max_bond = max_bond
        self.trunc_tol = trunc_tol
        self.tensors = [np.array([1, 0], dtype=complex).reshape(1, 2, 1)
                         for _ in range(n_qubits)]

    def apply_1q(self, gate, site):
        T = self.tensors[site]
        self.tensors[site] = np.einsum('ab,lbr->lar', gate, T)

    def apply_2q(self, gate, site):
        # gate acts on qubits (site, site + 1)
        A, B = self.tensors[site], self.tensors[site + 1]
        l, d1, m = A.shape
        m2, d2, r = B.shape

        theta = np.einsum('lam,mbr->labr', A, B)
        theta = theta.reshape(l * d1, d2 * r)
        g = gate.reshape(2, 2, 2, 2)
        theta = theta.reshape(l, d1, d2, r)
        theta = np.einsum('oOiI,liIr->loOr', g.reshape(2, 2, 2, 2), theta)
        theta = theta.reshape(l * 2, 2 * r)

        # split back into two tensors, keeping only significant singular values
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)
        keep = min(self.max_bond, np.sum(S > self.trunc_tol))
        keep = max(keep, 1)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]

        newA = U.reshape(l, 2, keep)
        newB = (np.diag(S) @ Vh).reshape(keep, 2, r)
        self.tensors[site], self.tensors[site + 1] = newA, newB

    def to_statevector(self):
        # only practical for small n
        vec = self.tensors[0].reshape(2, -1)
        for T in self.tensors[1:]:
            l, d, r = T.shape
            vec = vec @ T.reshape(l, d * r)
            vec = vec.reshape(-1, r)
        return vec.reshape(-1)

    def bond_dims(self):
        return [T.shape[2] for T in self.tensors[:-1]]

    def sample_probs(self):
        v = self.to_statevector()
        return np.abs(v) ** 2

    def apply_global(self, gate_matrix):
        # Applies a gate on all qubits by going through the full statevector.
        # Exact, but costs 2^n, so it's only used for things like Grover's
        # oracle and diffusion that entangle every qubit anyway.
        v = self.to_statevector()
        v = gate_matrix @ v
        self._from_statevector(v)

    def _from_statevector(self, v):
        # rebuild the MPS with a left-to-right sweep of SVDs
        n = self.n
        tensors = []
        remainder = v.reshape(1, -1)
        for site in range(n - 1):
            l, rest = remainder.shape
            remainder = remainder.reshape(l * 2, rest // 2)
            U, S, Vh = np.linalg.svd(remainder, full_matrices=False)
            keep = max(1, np.sum(S > self.trunc_tol))
            keep = min(keep, self.max_bond)
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
            tensors.append(U.reshape(l, 2, keep))
            remainder = (np.diag(S) @ Vh)
        tensors.append(remainder.reshape(remainder.shape[0], 2, 1))
        self.tensors = tensors


if __name__ == "__main__":
    print("=== Bell state (H + CNOT, 2 qubits) ===")
    mps = MPS(2)
    mps.apply_1q(H, 0)
    mps.apply_2q(CNOT, 0)
    print("statevector:", np.round(mps.to_statevector(), 3))
    print("bond dims  :", mps.bond_dims())

    print("\n=== 5-qubit GHZ state (H, then a chain of CNOTs) ===")
    n = 5
    mps = MPS(n)
    mps.apply_1q(H, 0)
    for i in range(n - 1):
        mps.apply_2q(CNOT, i)
    probs = mps.sample_probs()
    nz = [(format(i, f'0{n}b'), round(p, 3)) for i, p in enumerate(probs) if p > 1e-6]
    print("nonzero outcomes:", nz)
    print("bond dims        :", mps.bond_dims())

    print("\n=== CZ on |++> ===")
    mps = MPS(2)
    mps.apply_1q(H, 0)
    mps.apply_1q(H, 1)
    mps.apply_2q(CZ, 0)
    print("statevector:", np.round(mps.to_statevector(), 3), "(expected (1,1,1,-1)/2)")
