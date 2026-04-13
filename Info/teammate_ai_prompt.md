# Project Context for Claude — Citi Bike Overnight Rebalancing (Group 1)

## Project Overview

This is a DASC 32003 Spring 2026 optimization course project (Group 1: Abe Archer, Ethan Ericson, Trevor Bolena). We are implementing the overnight Citi Bike truck rebalancing model from Freund et al. (2020) "BSS Overnight" in Python using Gurobi.

The goal: route rebalancing trucks overnight (10pm–7am) across Midtown Manhattan Citi Bike stations to minimize user dissatisfaction over the coming day, by setting better starting bike counts at each station before morning demand begins.

---

## What Has Been Completed

### Data Pipeline (`Code/data_prep.py`)
The data preparation pipeline is complete and outputs two CSV files:

**`Code/Data/midtown_station_params.csv`** — Static parameters, one row per station (105 stations):
- `station_name`, `latitude`, `longitude`
- `kappa` — station capacity (max bikes it can hold), computed as `max(bikes + free)` across all January snapshots
- `mu_s_weekday`, `mu_s_weekend` — avg daytime departure rate (bikes/min) for weekdays and weekends
- `lambda_s_weekday`, `lambda_s_weekend` — avg daytime arrival rate (bikes/min) for weekdays and weekends

**`Code/Data/nightly_start_s.csv`** — Nightly starting conditions, one row per (station, night) (~3,255 rows):
- `date`, `station_name`, `start_s`
- `start_s` = bike count at the station at 10pm (last GBFS snapshot at or before 10pm that night)

### Key Data Decisions Made
- **Geographic scope**: Midtown Manhattan bounding box LAT 40.748–40.765, LON -74.012 to -73.960 (105 stations)
- **Capacity**: `max(bikes + free)` — NOT the `slots` field, which is unreliable due to disabled docks
- **Rates**: Per-day averaging (count trips per station per day, average across days, divide by 900 min) — more robust than dividing monthly totals by total monthly minutes
- **Weekday/weekend split**: Separate rates because commuter vs. leisure demand differs substantially
- **start_s**: Last snapshot at or before 10pm each night (not a monthly average) — trucks rebalance every night so we use actual nightly conditions
- **Cross-boundary trips included**: Rates reflect all activity at each station regardless of trip origin/destination

---

## Model Formulation

### Problem Structure
- Trucks operate overnight: 10pm to 7am
- T = 90 time steps (6 minutes each)
- γ = 7 bikes loaded/unloaded per time step
- K = number of trucks (start with K=1 for testing)
- Station set S, split into S+ (oversupplied) and S- (undersupplied)

### User Dissatisfaction Function (UDF)
Each station is modeled as an M/M/1/κ queue. Given x_0 bikes at a station at 7am (start of operating day), `F_s(x_0)` is the expected number of dissatisfied users over the coming day (7am–10pm).

Dissatisfaction occurs when:
- A user wants a bike but the station is empty (probability p_0, rate μ_s)
- A user wants to return a bike but the station is full (probability p_κ, rate λ_s)

The state probabilities p_i(t) (probability of i bikes at time t) evolve via **Kolmogorov Forward ODEs**:

```
dp_0/dt  = μ_s · p_1  - λ_s · p_0
dp_i/dt  = λ_s · p_{i-1} - (λ_s + μ_s) · p_i + μ_s · p_{i+1}   for 0 < i < κ
dp_κ/dt  = λ_s · p_{κ-1} - μ_s · p_κ
```

Initial condition: p_{x_0}(0) = 1, all other p_i(0) = 0.

Solved via **Euler's method** over 900 minutes (daytime window).

Dissatisfaction accumulated per minute:
```
dissatisfaction_rate(t) = μ_s · p_0(t) + λ_s · p_κ(t)
F_s(x_0) = Σ_t dissatisfaction_rate(t) · Δt
```

### Linearization Coefficient c_s
```
c_s = [F_s(start_s) - F_s(min_s)] / (start_s - min_s)
```
Where:
- `start_s` = bikes at station at 10pm (baseline if no trucks run)
- `min_s` = argmin F_s(x) over x ∈ [0, κ_s] (optimal starting count)
- `c_s` is **signed** — positive for oversupplied stations, negative for undersupplied

### IP Decision Variables
- `x_{stk} ∈ {0,1}` — is truck k at station s at time t?
- `y_{stk} ≥ 0` — bikes accessible to truck k at station s at time t
- `b_{tk} ≥ 0` — bikes on truck k at time t

### IP Objective
```
maximize Σ_{s,k} (y_{s,1,k} - y_{s,T,k}) · c_s
```
This rewards moving bikes away from oversupplied stations and toward undersupplied ones, weighted by how much each bike moved improves dissatisfaction.

### IP Constraints (from Freund et al. 2020)
1. **One location per truck**: Σ_s x_{stk} = 1 for all t, k
2. **Movement**: truck can stay or move to adjacent station each step
3. **Initialization**: trucks start at depot
4. **Pareto A**: y_{stk} ≤ κ_s · x_{stk} (can only access bikes if present)
5. **Pareto B**: net bikes moved at station s is bounded by improvement direction
6. **Conservation**: bikes accessible + bikes on truck are conserved
7. **Load when present**: can only load/unload γ bikes per step when at station
8. **Move OR load**: truck cannot move and load in the same time step

---

## Next Step: UDF Implementation

The next file to implement is `Code/udf.py`. It should:

1. **Load** `midtown_station_params.csv`
2. **For each station s**, for each integer x_0 ∈ [0, κ_s]:
   - Set initial condition: p_{x_0} = 1, all others 0
   - Run Euler's method on the Kolmogorov ODEs over 900 minutes (Δt = 1 min is fine)
   - At each step, accumulate `μ_s · p_0 + λ_s · p_κ` into F_s(x_0)
3. **Compute** `min_s = argmin F_s(x_0)` for each station
4. **Compute** `c_s = [F_s(start_s) - F_s(min_s)] / (start_s - min_s)` — note: when start_s == min_s, c_s = 0
5. **Output** a table with columns: `station_name, min_s, c_s` (and optionally the full F_s curve for analysis)

### Notes on UDF implementation
- Use weekday or weekend rates depending on which night you're solving — for now default to weekday
- κ_s varies per station (typically 15–40 docks), so the state space size varies
- Euler's method with Δt = 1 minute over 900 steps is sufficient
- The UDF needs to be recomputed per night only if rates change — since rates are static monthly averages, the F_s curves can be precomputed once and reused; only c_s changes nightly (because start_s changes)
- c_s = 0 when start_s == min_s (station is already optimal, no benefit to rebalancing)

---

## File Structure
```
Code/
  data_prep.py              — completed data pipeline
  udf.py                    — NEXT: implement this
  Data/
    midtown_station_params.csv   — 105 stations, static params
    nightly_start_s.csv          — ~3255 rows, one per (station, night)
    202501-citi-bike-nyc-stats.parquet
    202501-citibike-tripdata_1.csv
    202501-citibike-tripdata_2.csv
    202501-citibike-tripdata_3.csv
```

## Tech Stack
- Python, pandas, numpy
- Gurobi (gurobipy) for the IP solver — student license
- Paper reference: Freund et al. (2020) BSS Overnight
