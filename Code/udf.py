"""
User Dissatisfaction Function (UDF) computation for the overnight Citi Bike
rebalancing optimization.

This module computes f_s(x_0) — the expected number of dissatisfied users at
station s over the 900-minute operating day (7am-10pm), given that the station
starts the day with x_0 bikes. Each station is modeled as an M/M/1/kappa queue,
and f_s is computed by integrating the Kolmogorov Forward ODEs via Euler's
method.

From f_s we derive:
  - min_s    = argmin_{x_0} f_s(x_0)   — the optimal starting count
  - c_s      = [f_s(start_s) - f_s(min_s)] / (start_s - min_s)
               — signed linearization coefficient used as the objective
                 coefficient in the downstream Gurobi IP. Positive for
                 oversupplied stations, negative for undersupplied.

Inputs:
  - Data/midtown_station_params.csv   (static kappa, mu, lambda per station)
  - Data/nightly_start_s.csv          (per-night starting bike counts)

Outputs:
  - Data/station_udf.csv   — one row per (station_name, day_type, x_0)
                              columns: station_name, day_type, x_0, f, min_s, f_min
  - Data/nightly_cs.csv    — one row per (station_name, date)
                              columns: station_name, date, day_type, start_s,
                                       min_s, f_start, f_min, c_s

Math summary:
  dp_0/dt  = mu * p_1 - lam * p_0
  dp_i/dt  = lam * p_{i-1} - (lam + mu) * p_i + mu * p_{i+1}   (0 < i < kappa)
  dp_k/dt  = lam * p_{k-1} - mu * p_k
  Dissatisfaction rate r(t) = mu * p_0(t) + lam * p_kappa(t)
  f_s(x_0) = sum_{t=0}^{899} r(t) * dt       (left-Riemann, dt = 1 min)
"""

# Overload is just for type checking; allows multiple outputs types depending on input
from typing import Literal, overload

import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAMS_PATH = "Data/midtown_station_params.csv"
NIGHTLY_PATH = "Data/nightly_start_s.csv"
UDF_OUTPUT_PATH = "Data/station_udf.csv"
NIGHTLY_CS_OUTPUT_PATH = "Data/nightly_cs.csv"

DAY_MINUTES = 900  # 7am-10pm operating day
DT = 1.0  # Euler step size, in minutes

# Day-type column mapping: for each rate type, which CSV columns to read.
DAY_TYPES = [
    ("weekday", "mu_s_weekday", "lambda_s_weekday"),
    ("weekend", "mu_s_weekend", "lambda_s_weekend"),
]


# ---------------------------------------------------------------------------
# Core math functions
# ---------------------------------------------------------------------------


def build_generator(kappa: int, mu: float, lam: float) -> np.ndarray:
    """
    Build the (kappa+1, kappa+1) tridiagonal generator matrix q for an
    M/M/1/kappa queue.

    Entry q[i, j] is the rate at which probability flows from state j into
    state i. Reading column j: probability in state j drains out the main
    diagonal (negative) and flows into neighbor states.

    Structure:
      - Main diagonal:
          q[0, 0]         = -lam                (only outflow: a return,
                                                 rises from 0 to 1)
          q[i, i]         = -(lam + mu)         (interior: either a return
                                                 or pickup)
          q[kappa, kappa] = -mu                 (only outflow: a pickup,
                                                 drops from kappa to kappa-1)
      - Sub-diagonal  (one row below): lam  — successful returns push state up
      - Super-diagonal (one row above): mu  — successful pickups push state down

    Failed pickups (at state 0) and failed returns (at state kappa) are NOT in
    q — those events do not change the state, so they are only accumulated
    into the dissatisfaction vector, not the probability update.
    """

    # Create an empty matrix
    main_diag = np.empty(kappa + 1)
    main_diag[0] = -lam
    main_diag[-1] = -mu
    if kappa >= 2:
        main_diag[1:-1] = -(lam + mu)

    # Create other two diagonals
    # Sub-diagonal represents arrivals pushing state from state i-1 to i (down a column)
    # Sup-diagonal represents departures pushing you down from i + 1 to i (up a column)
    sub_diag = np.full(kappa, lam)  # q[i, i-1] = lam for i = 1 .. kappa
    sup_diag = np.full(kappa, mu)  # q[i, i+1] = mu  for i = 0 .. kappa-1

    # Combine three diagonals together
    q = np.diag(main_diag) + np.diag(sub_diag, k=-1) + np.diag(sup_diag, k=1)
    return q


# -----------------OVERLOAD TYPE PERMUTATIONS (Ignore)-----------------


@overload
def compute_f_curve(
    kappa: int,
    mu: float,
    lam: float,
    minutes: int = ...,
    dt: float = ...,
    *,
    return_final_p: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def compute_f_curve(
    kappa: int,
    mu: float,
    lam: float,
    minutes: int = ...,
    dt: float = ...,
    *,
    return_final_p: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...


# ---------------------------------------------------------------------


def compute_f_curve(
    kappa: int,
    mu: float,
    lam: float,
    minutes: int = DAY_MINUTES,
    dt: float = DT,
    *,
    return_final_p: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the full f_s curve for one station (one (kappa, mu, lam) triple).

    Vectorization trick: p is a (kappa+1, kappa+1) matrix where column j holds
    the probability distribution for the scenario that started with exactly j
    bikes. Initialized as the identity (each scenario is a certainty at t=0).
    One Euler step updates all kappa+1 scenarios simultaneously, so after
    `minutes` steps the returned f vector contains f_s(x_0) for every integer
    x_0 in [0, kappa] — the full curve in one pass.

    Euler step (per minute):
      1. Accumulate dissatisfaction for the current minute:
           f += dt * (mu * p[0, :] + lam * p[-1, :])
         (left-Riemann — read BEFORE stepping, matches spec Σ_t r(t)·Δt)
      2. Advance the distributions one minute forward:
           p = p + dt * (q @ p)

    Args:
      kappa: station capacity (number of docks).
      mu: per-minute pickup rate.
      lam: per-minute return rate.
      minutes: number of Euler steps (default 900 — the operating day).
      dt: Euler step size in minutes (default 1.0).
      return_final_p: if True, also return the final p matrix (for the
                      probability-conservation sanity check).

    Returns:
      f: np.ndarray of shape (kappa+1,). f[j] = f_s(j) = expected number of
         dissatisfied users over the day, given starting count j.
      p: (optional) final probability matrix after integration.
    """

    # Initialize generator matrix
    q = build_generator(kappa, mu, lam)
    # Initialize probability matrix
    p = np.eye(kappa + 1)  # column j = initial condition x_0 = j
    # Initialize f_curve array
    # At end of all steps, f[0] represents total disatisfaction if station started with 0 bikes
    f = np.zeros(kappa + 1)

    # Compute number of steps. Should be 900 since we set step = 1 minute
    num_steps = int(round(minutes / dt))
    for _ in range(num_steps):
        # Accumulate dissatisfaction BEFORE stepping p.
        # p[0, :]  = prob of empty, for every starting scenario simultaneously
        # p[-1, :] = prob of full,  for every starting scenario simultaneously
        f += dt * (mu * p[0, :] + lam * p[-1, :])
        # Euler step: all columns advance in parallel via one matrix multiply.
        p = p + dt * (q @ p)

    # Output final probability distribution if return_final_p = TRUE
    if return_final_p:
        return f, p
    return f


def compute_min_and_cs(f_curve: np.ndarray, start_s: int) -> tuple[int, float]:
    """
    Given a precomputed f_s curve and the station's starting bike count at 10pm,
    return (min_s, c_s).

    c_s is the signed slope of the chord from (start_s, f[start_s]) to
    (min_s, f[min_s]). Because f is convex:
      - start_s > min_s  =>  c_s > 0  (oversupplied: removing bikes helps)
      - start_s < min_s  =>  c_s < 0  (undersupplied: adding bikes helps)
      - start_s = min_s  =>  c_s = 0  (already optimal)

    The signed convention means the IP objective
      maximize Σ (y_{s,1,k} - y_{s,T,k}) * c_s
    correctly rewards movement in BOTH directions without any special-casing.
    """
    # Extract kappa from f_curve since we didn't pass it to the function
    kappa = len(f_curve) - 1
    # Defensive clip: start_s should always be in [0, kappa], but guard anyway.
    start_clipped = int(min(max(start_s, 0), kappa))

    # Finds the starting amount of bikes that produces the minimum f_curve value
    min_s = int(np.argmin(f_curve))
    # start_s (start_clipped) is how many bikes at a station at the BEGINNING of a night.
    # If a station already starts with the min_s (optimal) amount, no action is needed.
    # The per bike improvment in the objective thus becomes 0, since any movement from this
    # point isn't helpful.
    if start_clipped == min_s:
        c_s = 0.0
    # Otherwise, compute c_s as the slope of the chord betwen start_s and min_s
    else:
        c_s = float((f_curve[start_clipped] - f_curve[min_s]) / (start_clipped - min_s))
    return min_s, c_s


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _next_day_type(date: pd.Timestamp) -> str:
    """Return 'weekday' or 'weekend' for the day AFTER `date`.

    Trucks rebalance overnight to set conditions for the coming day. So the
    night of Sunday → Monday uses weekday rates (Monday's demand), and the
    night of Friday → Saturday uses weekend rates (Saturday's demand).
    """
    next_day = date + pd.Timedelta(days=1)
    return "weekend" if next_day.dayofweek >= 5 else "weekday"


def _main() -> None:

    # Grab start time here to measure how long computation takes
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Load inputs
    # -----------------------------------------------------------------------
    print("Loading station params and nightly start_s...")
    params = pd.read_csv(PARAMS_PATH)
    nightly = pd.read_csv(NIGHTLY_PATH, parse_dates=["date"])
    print(f"  Stations:        {len(params)}")
    print(f"  Nightly rows:    {len(nightly)}")

    # -----------------------------------------------------------------------
    # Step 2: Compute f_s curves for every (station, day_type) pair
    # -----------------------------------------------------------------------
    # f curves depend only on (kappa, mu, lam) — all static — so we compute
    # them once per (station, day_type). 105 stations x 2 day_types = 210 runs.
    # Results are collected into a tidy long-format DataFrame AND stashed in an
    # in-memory lookup dict so step 3 can join without re-reading a CSV.

    print("\nComputing f_s curves...")
    # Data structure that will store all udf's in long format.
    # Set up here to reduce computational cost of look ups
    # Will be output into csv for downstream ingestion.
    udf_records = []
    # f_curve lookup fuction for efficient lookups in this module.
    # Based on station and weekday/weekend, returns the entire f curve
    f_lookup: dict[tuple[str, str], np.ndarray] = {}
    # Stores warnings for instances where we have zero's for lambda or mu
    degenerate_warnings: list[str] = []

    # Retrieve all rows from parameter table and extract info
    for _, row in params.iterrows():
        station = row["station_name"]  # Get station name
        kappa = int(row["kappa"])  # Get capacity

        # Extract weekday/weekend specific rates
        for day_type, mu_col, lam_col in DAY_TYPES:
            mu = float(row[mu_col])
            lam = float(row[lam_col])

            # Compute_f_curve for one combo of kappa, mu, lam
            f = compute_f_curve(kappa, mu, lam)
            min_s = int(np.argmin(f))  # Find minimum starting number of bikes
            f_min = float(
                f[min_s]
            )  # Find total user disatisfaction for that optimal bike amount

            # Store that f in lookup table for future use
            f_lookup[(station, day_type)] = f

            # Check for degenerate conditions
            if mu == 0.0 or lam == 0.0:
                degenerate_warnings.append(
                    f"  {station} [{day_type}]: mu={mu:.4f}, lam={lam:.4f}, "
                    f"min_s={min_s}"
                )

            # Record UDF values in long format for downstream use outside this module
            for x_0 in range(kappa + 1):
                udf_records.append(
                    {
                        "station_name": station,
                        "day_type": day_type,
                        "x_0": x_0,
                        "f": float(f[x_0]),
                        "min_s": min_s,
                        "f_min": f_min,
                    }
                )

    station_udf = pd.DataFrame(udf_records)
    print(f"  f curves computed: {len(f_lookup)} (station, day_type) pairs")
    print(f"  Total f rows:      {len(station_udf)}")

    if degenerate_warnings:
        print(
            f"\n  WARNING: {len(degenerate_warnings)} (station, day_type) pairs "
            f"have mu=0 or lam=0:"
        )
        for w in degenerate_warnings[:10]:
            print(w)
        if len(degenerate_warnings) > 10:
            print(f"  ... and {len(degenerate_warnings) - 10} more")

    # -----------------------------------------------------------------------
    # Step 3: Compute per-night c_s by joining with nightly_start_s
    # -----------------------------------------------------------------------
    # c_s depends on start_s (which changes every night), so this must be
    # recomputed per (station, night). We pick the day_type based on the NEXT
    # day — Sunday night's truck run optimizes for Monday demand, so it uses
    # weekday rates.

    print("\nComputing nightly c_s...")
    # cs_records will hold nightly c_s for downstream use
    cs_records = []
    # Both of the follow are for tracking potential errors
    skipped_missing_f = 0
    start_clipped_count = 0

    # Take the nightly data and extract info to make per night calculations
    for _, row in nightly.iterrows():
        station = row["station_name"]
        date = row["date"]
        start_s_raw = row["start_s"]

        # Check if we've got data for start_s
        if pd.isna(start_s_raw):
            continue
        start_s = int(start_s_raw)

        # Compute the day type for the next day (what the night will prep for)
        day_type = _next_day_type(date)
        key = (station, day_type)
        # Check if we have data on that station for that day type. If not, skip
        if key not in f_lookup:
            skipped_missing_f += 1
            continue

        # Grab f_curve from lookup table
        f = f_lookup[key]
        kappa = len(f) - 1

        # Defensive clip: log any start_s outside [0, kappa]
        if start_s < 0 or start_s > kappa:
            start_clipped_count += 1
        start_s_clipped = int(min(max(start_s, 0), kappa))

        # Compute min_s and c_s
        min_s, c_s = compute_min_and_cs(f, start_s_clipped)

        # Append to records
        cs_records.append(
            {
                "station_name": station,
                "date": date.date(),
                "day_type": day_type,
                "start_s": start_s_clipped,
                "min_s": min_s,
                "f_start": float(f[start_s_clipped]),
                "f_min": float(f[min_s]),
                "c_s": c_s,
            }
        )

    # Create dataframe for future downstream use
    nightly_cs = pd.DataFrame(cs_records)
    print(f"  Nightly c_s rows:     {len(nightly_cs)}")
    if skipped_missing_f:
        print(f"  Rows skipped (no f): {skipped_missing_f}")
    if start_clipped_count:
        print(f"  Rows with start_s clipped: {start_clipped_count}")

    # -----------------------------------------------------------------------
    # Step 4: Sanity checks
    # -----------------------------------------------------------------------
    print("\n=== Sanity checks ===")

    # (1) Probability conservation — re-run 5 stations and verify column sums.
    # This resamples 5 stations and makes sure all probabilities add to 1
    print("(1) Probability conservation check (5 sample stations)...")
    sample_rows = params.sample(n=min(5, len(params)), random_state=0)
    max_drift = 0.0
    for _, row in sample_rows.iterrows():
        kappa = int(row["kappa"])
        mu = float(row["mu_s_weekday"])
        lam = float(row["lambda_s_weekday"])
        _, p_final = compute_f_curve(kappa, mu, lam, return_final_p=True)
        col_sums = p_final.sum(axis=0)
        drift = float(np.max(np.abs(col_sums - 1.0)))
        max_drift = max(max_drift, drift)
    print(f"    Max column-sum drift from 1.0: {max_drift:.2e}")
    assert (
        max_drift < 1e-4  # Makes sure drift is very small
    ), f"Probability drift {max_drift:.2e} too large — consider dt=0.5"

    # (2) Non-negativity of f
    print("(2) Non-negativity of f...")
    assert (station_udf["f"] >= 0).all(), "Negative f values detected"
    print(f"    OK — min f = {station_udf['f'].min():.6f}")

    # (3) Convexity (soft) — f_s should be convex, so second differences ≥ 0.
    print("(3) Convexity check (soft)...")
    convexity_violations = 0
    for (station, day_type), f in f_lookup.items():
        if len(f) < 3:
            continue
        d2 = np.diff(f, n=2)
        if (d2 < -1e-6).any():
            convexity_violations += 1
    print(
        f"    (station, day_type) pairs with convexity violations > 1e-6: "
        f"{convexity_violations} / {len(f_lookup)}"
    )

    # (4) Interior min for balanced stations (mu and lam within 2x of each other)
    # This checks that stations that have a similar rates of mu and lam bottom out
    # near the middle at the curve, not at the extremes.
    # This should always happen when mu and lam are similar.
    print("(4) Interior-min check for balanced stations...")
    interior_violations = 0
    for (station, day_type), f in f_lookup.items():
        p_row = params[params["station_name"] == station].iloc[0]
        mu = float(p_row[f"mu_s_{day_type}"])
        lam = float(p_row[f"lambda_s_{day_type}"])
        if mu <= 0 or lam <= 0:
            continue
        ratio = max(mu, lam) / min(mu, lam)
        if ratio > 2.0:
            continue
        min_s = int(np.argmin(f))
        kappa = len(f) - 1
        if min_s == 0 or min_s == kappa:
            interior_violations += 1
    print(f"    Balanced stations with endpoint min_s: {interior_violations}")

    # (5) c_s sign check
    print("(5) c_s sign consistency...")
    sign_violations = (
        (nightly_cs["start_s"] > nightly_cs["min_s"]) & (nightly_cs["c_s"] < 0)
    ).sum() + (
        (nightly_cs["start_s"] < nightly_cs["min_s"]) & (nightly_cs["c_s"] > 0)
    ).sum()
    assert sign_violations == 0, f"c_s sign violations: {sign_violations}"
    print("    OK — all signs consistent")

    # (6) Eyeball print — 5 stations covering different profiles
    # This just prints out some station stats to allow for us to eyeball it
    print("\n(6) Spot-check: 5 stations (weekday rates):")
    param_subset = params.copy()
    param_subset["mu_lam_ratio"] = param_subset["mu_s_weekday"] / (
        param_subset["lambda_s_weekday"] + 1e-12
    )
    picks = (
        pd.concat(
            [
                param_subset.nlargest(2, "mu_s_weekday"),  # high-mu (pickup heavy)
                param_subset.nlargest(2, "lambda_s_weekday"),  # high-lam (return heavy)
                param_subset.iloc[[len(param_subset) // 2]],  # middle-of-the-pack
            ]
        )
        .drop_duplicates("station_name")
        .head(5)
    )
    print(
        f"    {'station':<35} {'kap':>4} {'mu':>7} {'lam':>7} "
        f"{'min':>4} {'f[0]':>10} {'f[k]':>10}"
    )
    for _, row in picks.iterrows():
        station = row["station_name"]
        f = f_lookup[(station, "weekday")]
        kappa = int(row["kappa"])
        print(
            f"    {station[:35]:<35} {kappa:>4} "
            f"{row['mu_s_weekday']:>7.4f} {row['lambda_s_weekday']:>7.4f} "
            f"{int(np.argmin(f)):>4} {float(f[0]):>10.2f} {float(f[kappa]):>10.2f}"
        )

    # (7) Calendar boundary: print day_type for first week of January 2025
    print("\n(7) Calendar boundary check (first week of Jan 2025):")
    for d in pd.date_range("2025-01-01", "2025-01-08"):
        next_d = d + pd.Timedelta(days=1)
        print(
            f"    night of {d.date()} ({d.day_name()}) -> next day "
            f"{next_d.date()} ({next_d.day_name()}) -> {_next_day_type(d)}"
        )

    # -----------------------------------------------------------------------
    # Step 5: Summary and output
    # -----------------------------------------------------------------------
    print("\n=== Summary stats ===")
    print(f"  Station f rows:   {len(station_udf)}")
    print(f"  Nightly c_s rows: {len(nightly_cs)}")
    print(
        f"  c_s > 0 (oversupplied):  {(nightly_cs['c_s'] > 0).sum()}  "
        f"({(nightly_cs['c_s'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"  c_s < 0 (undersupplied): {(nightly_cs['c_s'] < 0).sum()}  "
        f"({(nightly_cs['c_s'] < 0).mean() * 100:.1f}%)"
    )
    print(
        f"  c_s = 0 (already opt):   {(nightly_cs['c_s'] == 0).sum()}  "
        f"({(nightly_cs['c_s'] == 0).mean() * 100:.1f}%)"
    )
    print(
        f"  |c_s| mean: {nightly_cs['c_s'].abs().mean():.4f}  "
        f"max: {nightly_cs['c_s'].abs().max():.4f}"
    )

    station_udf.to_csv(UDF_OUTPUT_PATH, index=False)
    print(f"\nSaved station f curves to {UDF_OUTPUT_PATH}")

    nightly_cs.to_csv(NIGHTLY_CS_OUTPUT_PATH, index=False)
    print(f"Saved nightly c_s to      {NIGHTLY_CS_OUTPUT_PATH}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    _main()
