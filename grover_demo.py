"""
Grover's search on the MPS simulator.

Finds one marked item among 2^n using about (pi/4) * sqrt(2^n) oracle calls.
The oracle and diffusion steps act on every qubit at once, so they're
applied as dense matrices with MPS.apply_global().
"""

import numpy as np
from mps_simulator import MPS, H


def grover_oracle(n, marked_index):
    # flips the sign of the marked state
    dim = 2 ** n
    O = np.eye(dim, dtype=complex)
    O[marked_index, marked_index] = -1
    return O


def grover_diffusion(n):
    # 2|s><s| - I, where |s> is the uniform superposition
    dim = 2 ** n
    s = np.ones((dim, 1)) / np.sqrt(dim)
    return 2 * (s @ s.T) - np.eye(dim)


def run_grover(n, marked_index, iterations=None):
    dim = 2 ** n
    if iterations is None:
        iterations = int(np.round(np.pi / 4 * np.sqrt(dim)))

    mps = MPS(n, max_bond=dim)  # no truncation, so the result is exact
    for q in range(n):
        mps.apply_1q(H, q)

    O = grover_oracle(n, marked_index)
    D = grover_diffusion(n)

    for it in range(iterations):
        mps.apply_global(O)
        mps.apply_global(D)

    probs = mps.sample_probs()
    return probs, iterations


if __name__ == "__main__":
    n = 4
    marked = 11
    probs, iters = run_grover(n, marked)

    print(f"Searching {2**n} items for index {marked}")
    print(f"Grover iterations used: {iters}  (~pi/4 * sqrt(N) = {np.pi/4*np.sqrt(2**n):.2f})")
    print(f"Probability of measuring the marked item: {probs[marked]:.4f}")
    print(f"(Random classical guess would give: {1/2**n:.4f})")

    top3 = np.argsort(probs)[::-1][:3]
    print("\nTop 3 most likely outcomes:")
    for idx in top3:
        print(f"  {format(idx, f'0{n}b')} (index {idx}): {probs[idx]:.4f}")
