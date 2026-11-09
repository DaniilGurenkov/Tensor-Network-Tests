"""
Matrix Product State (MPS) quantum circuit simulator.

Represents an n-qubit state as a chain of small tensors instead of one
2^n-dimensional vector. Two-qubit gates are applied by contracting
neighboring tensors and re-splitting them with SVD (truncating small
singular values) -- the same SVD math from our earlier conversation.

This lets you simulate many more qubits than a dense statevector approach,
as long as entanglement (bond dimension) stays manageable.
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
        """Start every qubit in |0>. Each tensor has shape (bond_left, 2, bond_right)."""
        self.n = n_qubits
        self.max_bond = max_bond
        self.trunc_tol = trunc_tol
        self.tensors = [np.array([1, 0], dtype=complex).reshape(1, 2, 1)
                         for _ in range(n_qubits)]

    # ---------- single-qubit gate ----------
    def apply_1q(self, gate, site):
        T = self.tensors[site]                       # (l, 2, r)
        self.tensors[site] = np.einsum('ab,lbr->lar', gate, T)

    # ---------- two-qubit gate on neighboring sites (site, site+1) ----------
    def apply_2q(self, gate, site):
        A, B = self.tensors[site], self.tensors[site + 1]
        l, d1, m = A.shape
        m2, d2, r = B.shape
        # Merge A and B into one big tensor: (l, d1, d2, r)
        theta = np.einsum('lam,mbr->labr', A, B)
        theta = theta.reshape(l * d1, d2 * r)
        # Apply gate (reshaped 4x4 acting on the two physical legs)
        g = gate.reshape(2, 2, 2, 2)                  # (out1,out2,in1,in2)
        theta = theta.reshape(l, d1, d2, r)
        theta = np.einsum('oOiI,liIr->loOr', g.reshape(2, 2, 2, 2), theta)
        theta = theta.reshape(l * 2, 2 * r)

        # SVD, then truncate small singular values (bond-dimension compression)
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)
        keep = min(self.max_bond, np.sum(S > self.trunc_tol))
        keep = max(keep, 1)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]

        newA = U.reshape(l, 2, keep)
        newB = (np.diag(S) @ Vh).reshape(keep, 2, r)
        self.tensors[site], self.tensors[site + 1] = newA, newB

    # ---------- reconstruct full statevector (only for small n / debugging) ----------
    def to_statevector(self):
        vec = self.tensors[0].reshape(2, -1)            # (2, r0)  [left bond=1 dropped]
        for T in self.tensors[1:]:
            l, d, r = T.shape
            vec = vec @ T.reshape(l, d * r)              # (prev_dim, d*r)
            vec = vec.reshape(-1, r)                      # (prev_dim*d, r)
        return vec.reshape(-1)

    def bond_dims(self):
        return [T.shape[2] for T in self.tensors[:-1]]

    def sample_probs(self):
        v = self.to_statevector()
        return np.abs(v) ** 2

    # ---------- apply an arbitrary N-qubit gate (exact, not MPS-compressed) ----------
    # NOTE: Grover/Shor spread entanglement across ALL qubits, so representing
    # them as a true low-bond-dimension MPS defeats the point. Here we drop back
    # to a dense statevector for the global gates (oracle, diffusion, QFT) --
    # this is exact but scales as 2^n, same as a normal quantum simulator.
    def apply_global(self, gate_matrix):
        v = self.to_statevector()
        v = gate_matrix @ v
        # re-decompose back into an MPS (sequential SVD sweep, as done by hand earlier)
        self._from_statevector(v)

    def _from_statevector(self, v):
        n = self.n
        tensors = []
        remainder = v.reshape(1, -1)  # (bond_left=1, rest)
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


# ---------------------------- demo ----------------------------
if __name__ == "__main__":
    print("=== Bell state via H + CNOT (2 qubits) ===")
    mps = MPS(2)
    mps.apply_1q(H, 0)
    mps.apply_2q(CNOT, 0)
    print("statevector:", np.round(mps.to_statevector(), 3))
    print("bond dims  :", mps.bond_dims())

    print("\n=== GHZ state on 5 qubits: H then CNOT chain ===")
    n = 5
    mps = MPS(n)
    mps.apply_1q(H, 0)
    for i in range(n - 1):
        mps.apply_2q(CNOT, i)
    probs = mps.sample_probs()
    nz = [(format(i, f'0{n}b'), round(p, 3)) for i, p in enumerate(probs) if p > 1e-6]
    print("nonzero outcomes:", nz)
    print("bond dims        :", mps.bond_dims())

    print("\n=== CZ entanglement check (matches our earlier CZ example) ===")
    mps = MPS(2)
    mps.apply_1q(H, 0)
    mps.apply_1q(H, 1)
    mps.apply_2q(CZ, 0)
    print("statevector:", np.round(mps.to_statevector(), 3), "-> should be (1,1,1,-1)/2")
