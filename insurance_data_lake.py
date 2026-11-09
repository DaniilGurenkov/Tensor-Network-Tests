"""
Tensor-train compression on a simulated insurance data lake.

Two layers:
  1. Raw tables (policies, claims). Individual records need to stay exact
     and rows aren't correlated with each other, so these stay as ordinary
     tables (Parquet with standard compression in a real lake).
  2. A benchmarking cube of aggregate metrics, shaped
     (region x coverage type x month x metric). Neighboring regions,
     coverage lines and months tend to move together, so the cube is close
     to low-rank and compresses well with a tensor-train (TT) decomposition.

All data here is randomly generated.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Raw tables
# ---------------------------------------------------------------------------

REGIONS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

COVERAGE_TYPES = [
    "auto", "home", "commercial_property", "general_liability",
    "workers_comp", "umbrella", "cyber", "professional_liability",
]

N_MONTHS = 60  # 5 years
METRICS = ["loss_ratio", "claim_frequency", "avg_premium",
           "claim_severity", "retention_rate"]


def generate_raw_tables(n_policies=40_000, n_claims=9_000, seed=0):
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
# 2. Benchmarking cube
# ---------------------------------------------------------------------------

def build_benchmark_cube(seed=0):
    # In practice this would come from aggregating the raw tables. Here it's
    # generated as a sum of a few outer products plus noise, so it has the
    # kind of cross-dimensional correlation real aggregate data has.
    rng = np.random.default_rng(seed)
    n_r, n_c, n_m, n_k = len(REGIONS), len(COVERAGE_TYPES), N_MONTHS, len(METRICS)

    n_factors = 6
    region_factors = rng.normal(size=(n_r, n_factors))
    coverage_factors = rng.normal(size=(n_c, n_factors))
    # random walk so adjacent months are correlated
    month_factors = np.cumsum(rng.normal(scale=0.15, size=(n_m, n_factors)), axis=0)
    metric_factors = rng.normal(size=(n_k, n_factors))

    cube = np.einsum("rf,cf,mf,kf->rcmk",
                      region_factors, coverage_factors,
                      month_factors, metric_factors)

    # rescale each metric to a realistic range and add a bit of noise
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
        # noise is scaled per metric so dollar-valued metrics don't swamp ratios
        slice_std = cube[:, :, :, k].std()
        cube[:, :, :, k] += rng.normal(scale=0.01 * slice_std,
                                        size=cube[:, :, :, k].shape)
    return cube


# ---------------------------------------------------------------------------
# 3. Tensor-train decomposition
# ---------------------------------------------------------------------------

def tt_svd(tensor, tol=1e-2):
    """
    TT-SVD: repeatedly reshape, take an SVD, keep U as the next core, and
    carry S @ Vt forward. tol is the overall relative error budget, split
    evenly across the d - 1 cuts.
    """
    dims = tensor.shape
    d = len(dims)
    eps = tol / np.sqrt(d - 1) if d > 1 else tol
    cores = []
    remainder = tensor.reshape(1, -1)
    bond_left = 1

    for i in range(d - 1):
        n_i = dims[i]
        rest_size = remainder.size // (bond_left * n_i)
        M = remainder.reshape(bond_left * n_i, rest_size)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # keep the smallest rank that retains (1 - eps^2) of the energy
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

    cores.append(remainder.reshape(bond_left, dims[-1], 1))
    return cores


def tt_reconstruct_clean(cores):
    # contract left to right, dropping the size-1 bonds at both ends
    result = cores[0][0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([result.ndim - 1], [0]))
    return result[..., 0]


def tt_balance_scale(cores):
    """
    Spread the overall scale evenly across the cores without changing the
    reconstructed tensor. TT-SVD leaves every core but the last orthonormal,
    so the last core carries all of the magnitude; its norm equals the norm
    of the whole tensor, and we redistribute that as a d-th root per core.
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
# 4. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1 — Raw tables")
    print("=" * 70)
    policies, claims = generate_raw_tables()
    print(f"policies table: {len(policies):,} rows x {policies.shape[1]} cols")
    print(policies.head(3).to_string(index=False))
    print(f"\nclaims table:   {len(claims):,} rows x {claims.shape[1]} cols")
    print(claims.head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 2 — Benchmarking cube (region x coverage x month x metric)")
    print("=" * 70)
    cube = build_benchmark_cube()
    print(f"cube shape: {cube.shape}  "
          f"({len(REGIONS)} regions x {len(COVERAGE_TYPES)} coverage types x "
          f"{N_MONTHS} months x {len(METRICS)} metrics)")
    raw_params = cube.size
    raw_bytes = cube.nbytes
    print(f"raw size: {raw_params:,} float64 values = {raw_bytes/1024:.1f} KB")

    print("\n" + "=" * 70)
    print("STEP 3 — Tensor-train compression")
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
    print("STEP 4 — Core shapes and a sample query")
    print("=" * 70)
    cores = tt_svd(cube, tol=0.02)
    cores = tt_balance_scale(cores)
    print("Core shapes (bond_left, physical_dim, bond_right):")
    labels = ["region", "coverage_type", "month", "metric"]
    for label, core in zip(labels, cores):
        print(f"  {label:14s}: {core.shape}")

    # loss ratio for CA auto policies over time
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
