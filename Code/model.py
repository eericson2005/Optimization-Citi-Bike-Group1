"""
Overnight Citi Bike truck rebalancing — Integer Program (Gurobi).

Routes K trucks across Midtown Manhattan stations over 90 six-minute time
steps (10pm–7am) to maximize total dissatisfaction reduction for the coming
day. Uses precomputed linearization coefficients c_s from Code/udf.py.

Decision variables (load/unload reformulation):
    x[s, t, k]  ∈ {0,1}   — truck k is at station s at time t
    l[s, t, k]  ≥ 0        — bikes loaded FROM station s by truck k at step t
    u[s, t, k]  ≥ 0        — bikes unloaded TO station s by truck k at step t
    b[t, k]     ≥ 0        — bikes on truck k at time t

Objective:
    maximize  Σ_s c_s · Σ_k Σ_t (l[s,t,k] − u[s,t,k])

Inputs:
    Data/midtown_station_params.csv   (station locations + kappa)
    Data/nightly_cs.csv               (per-night c_s, start_s, min_s)

Outputs:
    Data/rebalancing_results.csv      (per-station summary for solved night)
    Data/truck_routes.csv             (per-truck-step route & actions)
"""

import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAMS_PATH = "Data/midtown_station_params.csv"
NIGHTLY_CS_PATH = "Data/nightly_cs.csv"
RESULTS_PATH = "Data/rebalancing_results.csv"
ROUTES_PATH = "Data/truck_routes.csv"

T = 90                      # time steps (6 min each, 10pm–7am)
GAMMA = 7                   # max bikes loaded/unloaded per step
TRUCK_CAP = 25              # truck capacity in bikes
K = 3                       # number of trucks
ADJACENCY_RADIUS_M = 1000   # haversine meters for adjacency
DEFAULT_DATE = "2025-01-15"  # a Wednesday, for testing
TIME_LIMIT = 300             # Gurobi solve time limit (seconds)

EARTH_RADIUS_M = 6_371_000  # for haversine


# ---------------------------------------------------------------------------
# Haversine distance matrix
# ---------------------------------------------------------------------------

def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return S×S pairwise distance matrix in meters from lat/lon arrays."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)

    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :])
         * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Adjacency graph
# ---------------------------------------------------------------------------

def build_adjacency(
    params_df: pd.DataFrame,
    radius_m: float,
) -> dict[int, list[int]]:
    """Build adjacency dict from station coordinates.

    Returns mapping station_index -> list of neighbor indices (including self).
    Adds a virtual depot node (index S) connected to ALL stations and itself.
    """
    lats = params_df["latitude"].values
    lons = params_df["longitude"].values
    dist = haversine_matrix(lats, lons)

    S = len(params_df)
    adj: dict[int, list[int]] = {}

    for i in range(S):
        neighbors = list(np.where(dist[i] <= radius_m)[0])
        if i not in neighbors:
            neighbors.append(i)
        adj[i] = neighbors

    # Virtual depot connected to all stations and itself
    depot = S
    adj[depot] = list(range(S + 1))      # depot -> every station + self
    for i in range(S):
        adj[i].append(depot)              # every station -> depot

    degrees = [len(adj[i]) - 1 for i in range(S)]  # exclude self
    print(f"  Adjacency ({radius_m:.0f} m): "
          f"mean {np.mean(degrees):.1f}, "
          f"min {np.min(degrees)}, max {np.max(degrees)} neighbors")

    return adj


# ---------------------------------------------------------------------------
# Build Gurobi model
# ---------------------------------------------------------------------------

def build_model(
    night_df: pd.DataFrame,
    params_df: pd.DataFrame,
    adj: dict[int, list[int]],
    num_trucks: int,
    truck_cap: int,
    gamma: int,
    num_steps: int,
) -> tuple[gp.Model, dict]:
    """Construct the IP for a single night.

    Parameters
    ----------
    night_df : rows from nightly_cs.csv for one date, indexed the same as
               params_df (must be aligned by station index 0..S-1).
    params_df : midtown_station_params.csv loaded as DataFrame.
    adj : adjacency dict from build_adjacency (includes depot at index S).
    num_trucks, truck_cap, gamma, num_steps : model constants.

    Returns (model, var_dict) where var_dict has keys 'x', 'l', 'u', 'b'.
    """
    S = len(params_df)
    depot = S
    stations = list(range(S + 1))       # 0..S-1 real, S = depot
    real = list(range(S))
    trucks = list(range(num_trucks))
    times = list(range(num_steps))

    # Per-station data vectors (indexed 0..S-1)
    c_s = night_df["c_s"].values
    start_s = night_df["start_s"].values.astype(int)
    min_s = night_df["min_s"].values.astype(int)
    kappa = params_df["kappa"].values.astype(int)

    m = gp.Model("bike_rebalance")
    m.setParam("OutputFlag", 1)
    m.setParam("TimeLimit", TIME_LIMIT)

    # --- Variables --------------------------------------------------------
    x = m.addVars(stations, times, trucks, vtype=GRB.BINARY, name="x")
    l = m.addVars(real, times, trucks, lb=0.0, name="l")
    u = m.addVars(real, times, trucks, lb=0.0, name="u")
    b = m.addVars(times, trucks, lb=0.0, name="b")

    # --- Objective --------------------------------------------------------
    obj = gp.quicksum(
        c_s[s] * (l[s, t, k] - u[s, t, k])
        for s in real
        for t in times
        for k in trucks
    )
    m.setObjective(obj, GRB.MAXIMIZE)

    # --- Constraints ------------------------------------------------------

    # 1. One location per truck per step
    for t in times:
        for k in trucks:
            m.addConstr(
                gp.quicksum(x[s, t, k] for s in stations) == 1,
                name=f"one_loc_{t}_{k}",
            )

    # 2. Movement: can only be at s' at t+1 if was at a neighbor at t
    for t in range(num_steps - 1):
        for k in trucks:
            for sp in stations:
                m.addConstr(
                    x[sp, t + 1, k]
                    <= gp.quicksum(x[s, t, k] for s in adj[sp]),
                    name=f"move_{sp}_{t}_{k}",
                )

    # 3. Initialization: all trucks start at depot with 0 bikes
    for k in trucks:
        m.addConstr(x[depot, 0, k] == 1, name=f"init_loc_{k}")
        m.addConstr(b[0, k] == 0, name=f"init_bikes_{k}")

    # 4. Load/unload only if present at station
    for s in real:
        for t in times:
            for k in trucks:
                m.addConstr(l[s, t, k] <= gamma * x[s, t, k],
                            name=f"l_pres_{s}_{t}_{k}")
                m.addConstr(u[s, t, k] <= gamma * x[s, t, k],
                            name=f"u_pres_{s}_{t}_{k}")

    # 5. Rate limit per station per step
    for s in real:
        for t in times:
            for k in trucks:
                m.addConstr(l[s, t, k] + u[s, t, k] <= gamma,
                            name=f"rate_{s}_{t}_{k}")

    # 6. Move OR load: must also have been present at t-1
    for s in real:
        for t in range(1, num_steps):
            for k in trucks:
                m.addConstr(
                    l[s, t, k] + u[s, t, k] <= gamma * x[s, t - 1, k],
                    name=f"moveload_{s}_{t}_{k}",
                )

    # 7. Truck bike conservation
    for t in range(1, num_steps):
        for k in trucks:
            m.addConstr(
                b[t, k] == b[t - 1, k]
                + gp.quicksum(l[s, t, k] for s in real)
                - gp.quicksum(u[s, t, k] for s in real),
                name=f"conserve_{t}_{k}",
            )

    # 8. Truck capacity
    for t in times:
        for k in trucks:
            m.addConstr(b[t, k] <= truck_cap, name=f"cap_{t}_{k}")

    # 9. Station inventory bounds
    for s in real:
        net_removed = gp.quicksum(
            l[s, t, k] - u[s, t, k] for t in times for k in trucks
        )
        m.addConstr(net_removed <= start_s[s],
                     name=f"inv_floor_{s}")
        m.addConstr(-net_removed <= kappa[s] - start_s[s],
                     name=f"inv_ceil_{s}")

    # 10. Pareto B: directional + magnitude bound
    for s in real:
        delta = int(start_s[s]) - int(min_s[s])
        if delta > 0:  # oversupplied — only load
            for t in times:
                for k in trucks:
                    m.addConstr(u[s, t, k] == 0,
                                name=f"parB_noU_{s}_{t}_{k}")
            net_load = gp.quicksum(
                l[s, t, k] for t in times for k in trucks
            )
            m.addConstr(net_load <= delta, name=f"parB_cap_{s}")
        elif delta < 0:  # undersupplied — only unload
            for t in times:
                for k in trucks:
                    m.addConstr(l[s, t, k] == 0,
                                name=f"parB_noL_{s}_{t}_{k}")
            net_unload = gp.quicksum(
                u[s, t, k] for t in times for k in trucks
            )
            m.addConstr(net_unload <= -delta, name=f"parB_cap_{s}")
        else:  # already optimal — no transfers
            for t in times:
                for k in trucks:
                    m.addConstr(l[s, t, k] == 0,
                                name=f"parB_noL0_{s}_{t}_{k}")
                    m.addConstr(u[s, t, k] == 0,
                                name=f"parB_noU0_{s}_{t}_{k}")

    # 11. Symmetry-breaking: order trucks by first real station visited
    #     Introduce auxiliary f[k] = index of first real station truck k
    #     reaches, then f[0] <= f[1] <= f[2].
    f_var = m.addVars(trucks, lb=0, ub=S - 1, vtype=GRB.INTEGER, name="f")
    for k in trucks:
        # f[k] <= s + S*(1 - x[s,1,k]) for all real s
        # i.e. if truck k goes to station s at t=1, then f[k] <= s
        for s in real:
            m.addConstr(
                f_var[k] <= s + S * (1 - x[s, 1, k]),
                name=f"sym_f_{k}_{s}",
            )
    for k in range(num_trucks - 1):
        m.addConstr(f_var[k] <= f_var[k + 1], name=f"sym_ord_{k}")

    m.update()
    print(f"  Model: {m.NumVars:,} vars ({m.NumIntVars:,} integer), "
          f"{m.NumConstrs:,} constraints")

    return m, {"x": x, "l": l, "u": u, "b": b}


# ---------------------------------------------------------------------------
# Extract solution
# ---------------------------------------------------------------------------

def extract_solution(
    model: gp.Model,
    var_dict: dict,
    station_names: list[str],
    night_df: pd.DataFrame,
    num_trucks: int,
    num_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the solved model into route and results DataFrames."""
    x = var_dict["x"]
    l = var_dict["l"]
    u = var_dict["u"]
    b = var_dict["b"]
    S = len(station_names)
    depot = S

    # --- Truck routes -----------------------------------------------------
    route_rows = []
    for k in range(num_trucks):
        for t in range(num_steps):
            # Find station
            loc = depot
            for s in range(S + 1):
                if x[s, t, k].X > 0.5:
                    loc = s
                    break
            loc_name = station_names[loc] if loc < S else "__depot__"

            loaded = sum(l[s, t, k].X for s in range(S))
            unloaded = sum(u[s, t, k].X for s in range(S))
            bikes = b[t, k].X

            route_rows.append({
                "truck_id": k,
                "step": t,
                "station_name": loc_name,
                "bikes_loaded": round(loaded, 2),
                "bikes_unloaded": round(unloaded, 2),
                "bikes_on_truck": round(bikes, 2),
            })

    routes_df = pd.DataFrame(route_rows)

    # --- Per-station results ----------------------------------------------
    result_rows = []
    c_s = night_df["c_s"].values
    start_s = night_df["start_s"].values.astype(int)

    for s in range(S):
        net_removed = sum(
            l[s, t, k].X - u[s, t, k].X
            for t in range(num_steps)
            for k in range(num_trucks)
        )
        end_s = start_s[s] - round(net_removed)
        improvement = c_s[s] * net_removed

        result_rows.append({
            "station_name": station_names[s],
            "start_s": int(start_s[s]),
            "end_s": int(end_s),
            "net_change": round(net_removed, 2),
            "c_s": round(c_s[s], 6),
            "dissatisfaction_improvement": round(improvement, 4),
        })

    results_df = pd.DataFrame(result_rows)
    return results_df, routes_df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _main() -> None:
    t0 = time.time()
    solve_date = DEFAULT_DATE
    print(f"=== Bike Rebalancing IP — night of {solve_date} ===\n")

    # Step 1: Load data
    print("Step 1 — Loading data")
    params = pd.read_csv(PARAMS_PATH)
    nightly = pd.read_csv(NIGHTLY_CS_PATH, parse_dates=["date"])
    print(f"  {len(params)} stations, {len(nightly)} nightly rows")

    # Filter to selected night
    night_mask = nightly["date"] == pd.Timestamp(solve_date)
    night_df = nightly[night_mask].copy()
    night_df = night_df.merge(
        params[["station_name"]].reset_index(),
        on="station_name",
        how="inner",
    )
    night_df = night_df.sort_values("index").reset_index(drop=True)

    if len(night_df) != len(params):
        missing = set(params["station_name"]) - set(night_df["station_name"])
        print(f"  WARNING: {len(missing)} stations missing for {solve_date}")
        # Align params to match
        params = params[params["station_name"].isin(night_df["station_name"])]
        params = params.reset_index(drop=True)
        night_df = night_df.reset_index(drop=True)

    station_names = list(params["station_name"])
    print(f"  Night of {solve_date}: {len(night_df)} stations, "
          f"c_s range [{night_df['c_s'].min():.4f}, {night_df['c_s'].max():.4f}]")

    # Step 2: Build adjacency
    print("\nStep 2 — Building adjacency graph")
    adj = build_adjacency(params, ADJACENCY_RADIUS_M)

    # Step 3: Build and solve model
    print(f"\nStep 3 — Building model (K={K}, T={T}, gamma={GAMMA})")
    model, var_dict = build_model(
        night_df, params, adj, K, TRUCK_CAP, GAMMA, T,
    )

    print("\nSolving...")
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print(f"\n  OPTIMAL — objective = {model.ObjVal:.4f}")
    elif model.Status == GRB.TIME_LIMIT:
        print(f"\n  TIME LIMIT — best objective = {model.ObjVal:.4f}, "
              f"gap = {model.MIPGap:.2%}")
    else:
        print(f"\n  Solve status: {model.Status}")
        return

    # Step 4: Extract and print solution
    print("\nStep 4 — Extracting solution")
    results_df, routes_df = extract_solution(
        model, var_dict, station_names, night_df, K, T,
    )

    # Summary
    touched = results_df[results_df["net_change"].abs() > 0.01]
    total_loaded = results_df[results_df["net_change"] > 0]["net_change"].sum()
    total_unloaded = -results_df[results_df["net_change"] < 0]["net_change"].sum()
    total_improvement = results_df["dissatisfaction_improvement"].sum()

    print(f"\n  Stations touched: {len(touched)} / {len(results_df)}")
    print(f"  Bikes loaded (from oversupplied):  {total_loaded:.0f}")
    print(f"  Bikes unloaded (to undersupplied): {total_unloaded:.0f}")
    print(f"  Total dissatisfaction improvement:  {total_improvement:.4f}")

    # Truck route summary
    for k in range(K):
        truck_route = routes_df[routes_df["truck_id"] == k]
        visited = truck_route[truck_route["station_name"] != "__depot__"]
        unique_stations = visited["station_name"].nunique()
        total_l = truck_route["bikes_loaded"].sum()
        total_u = truck_route["bikes_unloaded"].sum()
        print(f"  Truck {k}: {unique_stations} stations, "
              f"loaded {total_l:.0f}, unloaded {total_u:.0f}")

    # Step 5: Save outputs
    print("\nStep 5 — Saving results")
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"  Saved {RESULTS_PATH} ({len(results_df)} rows)")
    routes_df.to_csv(ROUTES_PATH, index=False)
    print(f"  Saved {ROUTES_PATH} ({len(routes_df)} rows)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    _main()
