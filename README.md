# Tensor Network Experiments

Small NumPy experiments with tensor networks: a matrix product state (MPS) quantum circuit simulator, Grover's search run on that simulator, and tensor-train (TT) compression applied to a simulated insurance dataset.

## Files

| File | What it does |
|---|---|
| `mps_simulator.py` | MPS simulator. Stores an n-qubit state as a chain of small tensors. Two-qubit gates are applied by contracting neighboring tensors and re-splitting them with an SVD, dropping small singular values. |
| `grover_demo.py` | Grover's search on the MPS simulator. The oracle and diffusion steps act on all qubits, so they are applied as dense matrices. |
| `insurance_data_lake.py` | TT-SVD compression of a 50 × 8 × 60 × 5 benchmarking cube (region × coverage type × month × metric). All data is randomly generated. |

## Requirements

Python 3 with NumPy and pandas:

    pip install numpy pandas

## Running

    python mps_simulator.py
    python grover_demo.py
    python insurance_data_lake.py

## Results

**MPS simulator.** Builds a Bell state and a 5-qubit GHZ state correctly. The bond dimension stays at 2 across the GHZ chain.

**Grover's search.** With 16 items and 3 iterations, the marked item is measured with probability 0.961. A random guess would succeed with probability 0.0625.

**Tensor-train compression** of the 120,000-value cube:

| Tolerance | TT-ranks | Parameters | Compression | Relative error |
|---|---|---|---|---|
| 0.1 | [2, 1, 1] | 181 | 663× | 5.8% |
| 0.05 | [3, 2, 1] | 323 | 372× | 4.0% |
| 0.01 | [5, 5, 2] | 1,060 | 113× | 0.57% |
| 0.001 | [24, 13, 2] | 5,266 | 23× | 0.08% |

## Notes

- `MPS.apply_global()` goes through the full statevector, so Grover's search still costs O(2^n) memory here. Tensor networks help most when entanglement stays low, and Grover's search spreads entanglement across every qubit.
- The raw policy and claim tables in `insurance_data_lake.py` are not compressed. Individual records need to stay exact, and the rows aren't correlated with each other, so a low-rank structure doesn't help.
