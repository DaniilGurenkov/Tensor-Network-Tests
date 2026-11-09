"""
Example data lake for a mid-size insurance brokerage, plus a tensor-train
(TT) compressed benchmarking cube built on top of it.

This follows directly from the earlier discussion: a data lake has two very
different kinds of content, and only one of them is a good fit for tensor
compression.

  1. RAW TABULAR LAYER (policies, claims) — relational, sparse, individually
     important records (a specific claim amount, a specific policyholder's
     deductible). This stays as ordinary tables — in a real lake you'd store
     these as Parquet/ORC with standard columnar compression (Snappy/Zstd).
     No tensor decomposition here; there's no "bond dimension" to exploit
     across unrelated policy rows, and you can't afford to lose exact figures
     to lossy compression on individual records.

  2. DERIVED BENCHMARKING CUBE — a dense, multi-dimensional array of
     aggregate metrics: (region x coverage_type x month x metric). This is
     exactly the "dense array with correlated structure across dimensions"
     case where tensor-train decomposition earns its keep — nearby regions
     behave similarly, adjacent months trend smoothly, so the cube is
     naturally low-rank even though it's stored densely.

The TT decomposition here is the exact same algorithm as the MPS
construction we built for GHZ earlier: reshape -> SVD -> peel off a core ->
carry the remainder forward -> repeat. The only differences from the
quantum case:
  - values are real numbers (loss ratios, dollars), not complex amplitudes
  - truncation uses a numerical tolerance (relative error budget) instead of
    exact zero/nonzero singular values, since real-world data doesn't
    factor perfectly
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Raw tabular layer — policies and claims (stays as ordinary tables)
# ---------------------------------------------------------------------------

REGIONS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]  # 50 regions — typical footprint for a mid-size national brokerage

COVERAGE_TYPES = [
    "auto", "home", "commercial_property", "general_liability",
    "workers_comp", "umbrella", "cyber", "professional_liability",
]  # 8 coverage lines

N_MONTHS = 60  # 5 years of monthly benchmarking history
METRICS = ["loss_ratio", "claim_frequency", "avg_premium",
           "claim_severity", "retention_rate"]


def generate_raw_tables(n_policies=40_000, n_claims=9_000, seed=0):
    """Simulate the kind of relational tables that actually sit in the
    data lake's tabular layer — individual policy and claim records."""
    rng = np.random.default_rng(seed)

    policies = pd.DataFrame({
        "policy_id": np.arange(n_policies),
        "region": rng.choice(REGIONS, n_policies),
        "coverage_type": rng.choice(COVERAGE_TYPES, n_policies),
        "effective_month": rng.integers(0, N_MONTHS, n_policies),
        "premium": np.round(rng.gamma(shape=4.0, scale=350, size=n_policies), 2),
        "deductible": rng.choice([250, 500, 1000, 2500, 5000], n_policies),
    })

    claims = pd.DataFrame({
        "claim_id": np.arange(n_claims),
        "policy_id": rng.integers(0, n_policies, n_claims),
        "claim_month": rng.integers(0, N_MONTHS, n_claims),
        "claim_amount": np.round(rng.gamma(shape=2.0, scale=2500, size=n_claims), 2),
        "status": rng.choice(["open", "closed", "denied"], n_claims,
                              p=[0.15, 0.75, 0.10]),
    })

    return policies, claims


# ---------------------------------------------------------------------------
# 2. Derived benchmarking cube — the tensor-compressible layer
# ---------------------------------------------------------------------------

def build_benchmark_cube(seed=0):
    """
    Build the (region x coverage_type x month x metric) cube directly from a
    low-rank generative model — a stand-in for what you'd normally get by
    aggregating the raw tables above with a GROUP BY. Generating it as a sum
    of a few outer products (plus noise) is what gives it genuine cross-
    dimensional correlation, the same property that gave low bond dimension
    in the quantum examples: nearby regions/coverage types/months behave
    similarly rather than varying independently.
    """
    rng = np.random.default_rng(seed)
    n_r, n_c, n_m, n_k = len(REGIONS), len(COVERAGE_TYPES), N_MONTHS, len(METRICS)

    # A handful of latent "market factors" (rank-6ish), each factor smoothly
    # varying across region / coverage / time / metric — this is what a real
    # book of business looks like: a few underlying drivers (regional risk
    # level, coverage-line profitability, seasonal trend, metric scale)
    # rather than 144,000 independent random numbers.
    n_factors = 6
    region_factors = rng.normal(size=(n_r, n_factors))
    coverage_factors = rng.normal(size=(n_c, n_factors))
    # smooth time trend per factor (cumulative sum of small steps = smooth,
    # not white noise) so adjacent months correlate, like real trends do
    month_factors = np.cumsum(rng.normal(scale=0.15, size=(n_m, n_factors)), axis=0)
    metric_factors = rng.normal(size=(n_k, n_factors))

    cube = np.einsum("rf,cf,mf,kf->rcmk",
                      region_factors, coverage_factors,
                      month_factors, metric_factors)

    # rescale each metric slice to a realistic range, then add a little
    # measurement noise (this is what keeps the cube from compressing to
    # a perfectly trivial rank-1 result — realistic data is low-rank, not
    # zero-rank)
    ranges = {
        "loss_ratio": (0.4, 1.1),
        "claim_frequency": (0.01, 0.12),
        "avg_premium": (600, 2200),
        "claim_severity": (1500, 9000),
        "retention_rate": (0.75, 0.97),
    }
    cube = cube - cube.min()
    cube = cube / cube.max()
    for k, metric in enumerate(METRICS):
        lo, hi = ranges[metric]
        cube[:, :, :, k] = lo + cube[:, :, :, k] * (hi - lo)
        # noise scaled to THIS metric's own magnitude — using the global
        # cube std here (mixing dollars with ratios) was the bug: it let
        # noise sized for claim_severity (~1500-9000) swamp loss_ratio
        # (~0.4-1.1), corrupting it far outside any realistic range.
        slice_std = cube[:, :, :, k].std()
        cube[:, :, :, k] += rng.normal(scale=0.01 * slice_std,
                                        size=cube[:, :, :, k].shape)

    return cube


# ---------------------------------------------------------------------------
# 3. Tensor-Train decomposition — same algorithm as the MPS/GHZ sweep
# ---------------------------------------------------------------------------

def tt_svd(tensor, tol=1e-2):
    """
    Decompose an n-dimensional array into a chain of 3-index TT cores, via
    the exact sequential SVD sweep used for the GHZ-state MPS earlier:
    reshape -> SVD -> peel a core off the left -> carry Sigma*V^T forward as
    the remainder -> repeat on what's left.

    tol is a relative Frobenius-norm error budget per cut (spent evenly
    across all cuts), analogous to truncating small singular values in the
    Grover diffusion / QFT examples, except here it's a numerical tolerance
    rather than an exact zero.
    """
    dims = tensor.shape
    d = len(dims)
    eps = tol / np.sqrt(d - 1) if d > 1 else tol  # split error budget across cuts

    cores = []
    remainder = tensor.reshape(1, -1)  # (bond_left=1, rest) to start
    bond_left = 1

    for i in range(d - 1):
        n_i = dims[i]
        rest_size = remainder.size // (bond_left * n_i)
        M = remainder.reshape(bond_left * n_i, rest_size)

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # truncate: keep enough singular values to hold (1 - eps^2) of the
        # squared Frobenius norm — same "keep the dominant terms, drop the
        # small ones" logic as SVD compression/low-rank approximation
        total_energy = np.sum(S ** 2)
        if total_energy > 0:
            cum_energy = np.cumsum(S ** 2) / total_energy
            r = int(np.searchsorted(cum_energy, 1 - eps ** 2) + 1)
            r = max(1, min(r, len(S)))
        else:
            r = 1

        core = U[:, :r].reshape(bond_left, n_i, r)
        cores.append(core)

        remainder = (np.diag(S[:r]) @ Vt[:r, :])
        bond_left = r

    # last core absorbs whatever remains
    cores.append(remainder.reshape(bond_left, dims[-1], 1))
    return cores


def tt_reconstruct_clean(cores):
    """Simpler, robust version: contract left to right, squeezing the
    dummy size-1 bonds at the very start and end (same dummy-leg convention
    as the edge tensors A and C in the GHZ example)."""
    result = cores[0][0]  # drop the leading dummy bond -> shape (n0, r0)
    for core in cores[1:]:
        # result: (..., r_prev); core: (r_prev, n_i, r_next)
        result = np.tensordot(result, core, axes=([result.ndim - 1], [0]))
    return result[..., 0]  # drop the trailing dummy bond


def tt_balance_scale(cores):
    """
    Redistribute magnitude evenly across TT cores without changing the
    reconstructed tensor at all.

    Why this is possible: contraction of the TT chain is linear in each
    individual core (holding the others fixed) — exactly the same
    multilinearity that lets you freely move a scalar between the bra and
    ket of an inner product. So if you scale core i by alpha and core j by
    1/alpha, the reconstructed tensor is completely unchanged; you're just
    moving where the "size" of the answer is bookkept.

    tt_svd as written always leaves every core except the last as a left-
    orthonormal isometry (U from an SVD, columns satisfying U^T U = I), and
    dumps 100% of the tensor's actual scale into the final core — that's
    why claim_severity/avg_premium (large numbers) blew the last core's
    entries up into the hundreds of thousands while the region/coverage/
    month cores stayed near +-0.1 to 0.3. Since those isometries preserve
    Frobenius norm exactly at each step, the norm of the *last* core alone
    equals the norm of the whole (truncated) tensor — so it's a clean,
    exact "grand total" to redistribute as the d-th root across all d cores.
    """
    d = len(cores)
    grand_scale = np.linalg.norm(cores[-1])
    if grand_scale == 0:
        return cores
    factor = grand_scale ** (1.0 / d)

    balanced = []
    for i, core in enumerate(cores):
        if i < d - 1:
            balanced.append(core * factor)
        else:
            balanced.append(core / grand_scale * factor)
    return balanced


def tt_num_params(cores):
    return sum(c.size for c in cores)


# ---------------------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1 — Raw tabular layer (stays as ordinary tables in the lake)")
    print("=" * 70)
    policies, claims = generate_raw_tables()
    print(f"policies table: {len(policies):,} rows x {policies.shape[1]} cols")
    print(policies.head(3).to_string(index=False))
    print(f"\nclaims table:   {len(claims):,} rows x {claims.shape[1]} cols")
    print(claims.head(3).to_string(index=False))
    print("\n-> These stay relational: Parquet/columnar storage, standard")
    print("   compression (Snappy/Zstd). No tensor decomposition — individual")
    print("   records matter and there's no cross-row correlation structure")
    print("   to exploit the way there is in the aggregate cube below.")

    print("\n" + "=" * 70)
    print("STEP 2 — Derived benchmarking cube (region x coverage x month x metric)")
    print("=" * 70)
    cube = build_benchmark_cube()
    print(f"cube shape: {cube.shape}  "
          f"({len(REGIONS)} regions x {len(COVERAGE_TYPES)} coverage types x "
          f"{N_MONTHS} months x {len(METRICS)} metrics)")
    raw_params = cube.size
    raw_bytes = cube.nbytes
    print(f"raw size: {raw_params:,} float64 values = {raw_bytes/1024:.1f} KB")

    print("\n" + "=" * 70)
    print("STEP 3 — Tensor-Train compression of the cube")
    print("=" * 70)
    for tol in (0.10, 0.05, 0.01, 0.001):
        cores = tt_svd(cube, tol=tol)
        ranks = [c.shape[0] for c in cores[1:]]
        compressed_params = tt_num_params(cores)
        reconstructed = tt_reconstruct_clean(cores)
        rel_err = np.linalg.norm(reconstructed - cube) / np.linalg.norm(cube)
        ratio = raw_params / compressed_params
        print(f"tol={tol:<6} TT-ranks={ranks}  "
              f"params: {compressed_params:,} (vs {raw_params:,})  "
              f"compression: {ratio:5.2f}x  "
              f"actual rel. error: {rel_err:.4f}")

    print("\n" + "=" * 70)
    print("STEP 4 — Inspect one core, and a benchmark query on the compressed form")
    print("=" * 70)
    cores = tt_svd(cube, tol=0.02)
    cores = tt_balance_scale(cores)
    print("Core shapes (bond_left, physical_dim, bond_right):")
    labels = ["region", "coverage_type", "month", "metric"]
    for label, core in zip(labels, cores):
        print(f"  {label:14s}: {core.shape}")

    # Example benchmarking query: average loss_ratio for CA, auto, over time
    # — reconstruct just the slice you need rather than the whole cube.
    full = tt_reconstruct_clean(cores)
    ca_idx = REGIONS.index("CA")
    auto_idx = COVERAGE_TYPES.index("auto")
    lr_idx = METRICS.index("loss_ratio")
    series = full[ca_idx, auto_idx, :, lr_idx]
    print(f"\nCA / auto loss_ratio, first 6 months (from compressed reconstruction):")
    print(np.round(series[:6], 3))
    original_series = cube[ca_idx, auto_idx, :, lr_idx]
    print(f"CA / auto loss_ratio, first 6 months (original):")
    print(np.round(original_series[:6], 3))
