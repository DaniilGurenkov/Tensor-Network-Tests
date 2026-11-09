"""
Grover's algorithm, run on the MPS simulator.

Search an unsorted list of 2^n items for one "marked" item using
O(sqrt(2^n)) oracle calls instead of O(2^n) classically.

Because the oracle and diffusion operator entangle ALL qubits together
(not just neighbors), we apply them as dense global gates via
MPS.apply_global() -- exact, but scales as 2^n, same as any statevector
simulator. This is the real reason tensor-network compression doesn't
give you a shortcut for these algorithms: Grover's/Shor's *are* the
high-entanglement circuits that MPS compression can't help with.
"""

import numpy as np
from mps_simulator import MPS, H

def grover_oracle(n, marked_index):
    """Flip the sign of the marked basis state, identity elsewhere."""
    dim = 2 ** n
    O = np.eye(dim, dtype=complex)
    O[marked_index, marked_index] = -1
    return O

def grover_diffusion(n):
    """2|s><s| - I, where |s> is the equal superposition state."""
    dim = 2 ** n
    s = np.ones((dim, 1)) / np.sqrt(dim)
    return 2 * (s @ s.T) - np.eye(dim)

def run_grover(n, marked_index, iterations=None):
    dim = 2 ** n
    if iterations is None:
        iterations = int(np.round(np.pi / 4 * np.sqrt(dim)))

    mps = MPS(n, max_bond=dim)  # allow full bond dim -- exact simulation
    for q in range(n):
        mps.apply_1q(H, q)  # equal superposition

    O = grover_oracle(n, marked_index)
    D = grover_diffusion(n)

    for it in range(iterations):
        mps.apply_global(O)
        mps.apply_global(D)

    probs = mps.sample_probs()
    return probs, iterations

if __name__ == "__main__":
    n = 4                      # 16 items -- small enough to verify exactly
    marked = 11                 # the "answer" we're searching for
    probs, iters = run_grover(n, marked)

    print(f"Searching {2**n} items for index {marked}")
    print(f"Grover iterations used: {iters}  (~pi/4 * sqrt(N) = {np.pi/4*np.sqrt(2**n):.2f})")
    print(f"Probability of measuring the marked item: {probs[marked]:.4f}")
    print(f"(Random classical guess would give: {1/2**n:.4f})")

    top3 = np.argsort(probs)[::-1][:3]
    print("\nTop 3 most likely outcomes:")
    for idx in top3:
        print(f"  {format(idx, f'0{n}b')} (index {idx}): {probs[idx]:.4f}")
