# Teammate Explainer — Citi Bike Overnight Rebalancing Project

**Group 1 | DASC 32003 Spring 2026 | Abe Archer, Ethan Ericson, Trevor Bolena**

---

## 1. What Are We Doing?

We are building a system that tells rebalancing trucks where to go overnight (10pm–7am) to move bikes between Citi Bike stations in Midtown Manhattan. The goal is to minimize how many users are dissatisfied the next day — either because a station ran out of bikes, or because a station was too full to return one.

The project is based on the paper: **Freund et al. (2020) "BSS Overnight"**.

---

## 2. The Big Picture

### Why does rebalancing matter?
Bike-share stations are constantly getting out of balance. Heavy commuter flows drain bikes from residential areas in the morning and fill up office-area stations. By the end of the day, some stations are overflowing and others are empty. Trucks drive around overnight to reset things for the next morning.

### What are we optimizing?
We want truck routes that maximize improvement in user satisfaction across all 105 Midtown stations, given:
- A fixed number of trucks (K)
- 90 time steps overnight (6 minutes each, 10pm–7am)
- Trucks can load/unload up to γ = 7 bikes per time step
- Trucks can only be at one station at a time

---

## 3. Key Concepts

### 3.1 The Operating Day vs. The Overnight Window

There are two distinct time windows:

| Window | Hours | Purpose |
|---|---|---|
| **Daytime** | 7am – 10pm | Users ride bikes; dissatisfaction accumulates |
| **Overnight** | 10pm – 7am | Trucks rebalance; no user demand |

Trucks work overnight to **set the starting conditions** for the next day. The number of bikes at a station at **7am** (after trucks finish) determines how much dissatisfaction occurs during the day.

### 3.2 Station Parameters

For each of the 105 Midtown stations we track:

- **κ_s (kappa)** — station capacity: the maximum number of bikes it can hold. Computed as `max(bikes + free docks)` observed across January. We don't use the `slots` field from the raw data because ~87% of records show discrepancies caused by disabled docks.

- **μ_s (mu)** — departure rate: average number of bikes picked up per minute during the day (7am–10pm). Represents users taking bikes.

- **λ_s (lambda)** — arrival rate: average number of bikes returned per minute during the day. Represents users returning bikes.

- **start_s** — the actual number of bikes at the station at **10pm** on a given night. This is the state the trucks inherit. It changes every night, so we store one value per station per night.

### 3.3 How μ_s and λ_s Are Calculated

For each station, for each day type (weekday or weekend):
1. Count trips departing/arriving at that station on each observed day
2. Average those daily counts across all days of that type in January
3. Divide by 900 (the number of daytime minutes) to get trips/minute

We compute **separate weekday and weekend rates** because commuter demand (Mon–Fri) differs substantially from leisure demand (Sat–Sun). When solving the IP for a given night, we use the rate matching the coming day — e.g., Sunday night uses weekend rates for Monday.

### 3.4 The Queue Model

Each station is modeled as an **M/M/1/κ queue** (a classic queueing theory model):
- Bikes arrive at rate λ_s (drop-offs) and depart at rate μ_s (pickups)
- The station holds at most κ_s bikes
- State i = number of bikes currently at the station

Two bad events cause dissatisfaction:
- **Station empty (state 0)**: a user wants a bike but can't get one
- **Station full (state κ)**: a user wants to return a bike but can't

### 3.5 The User Dissatisfaction Function (UDF)

`F_s(x_0)` is the **expected number of dissatisfied users** over the full day (7am–10pm) if the station starts with `x_0` bikes at 7am.

It is computed by solving differential equations (Kolmogorov Forward ODEs) that describe how the probability distribution over bike counts evolves over time.

**The ODEs:**
```
dp_0/dt  = μ_s · p_1  - λ_s · p_0
dp_i/dt  = λ_s · p_{i-1} - (λ_s + μ_s) · p_i + μ_s · p_{i+1}   (0 < i < κ)
dp_κ/dt  = λ_s · p_{κ-1} - μ_s · p_κ
```

Where `p_i(t)` = probability the station has exactly i bikes at time t.

**Starting condition**: `p_{x_0}(0) = 1`, all others 0 (we know exactly how many bikes are there at 7am).

We solve these numerically using **Euler's method** over 900 minutes with Δt = 1 minute.

At each minute, the dissatisfaction rate is:
```
μ_s · p_0(t)   +   λ_s · p_κ(t)
(failed pickups)   (failed returns)
```

Summing this over all 900 minutes gives `F_s(x_0)`.

We compute `F_s(x_0)` for every integer x_0 from 0 to κ_s, giving a full curve per station.

### 3.6 The Linearization Coefficient c_s

`c_s` summarizes how much dissatisfaction changes per bike moved at station s. It is the slope of the line from the station's current state to its optimal state on the F_s curve.

```
c_s = [F_s(start_s) - F_s(min_s)] / (start_s - min_s)
```

Where:
- `start_s` = bikes at 10pm (what happens if no trucks come)
- `min_s` = the x_0 that minimizes F_s (the optimal number of bikes for 7am)
- If start_s == min_s, the station is already optimal → c_s = 0

**The sign of c_s matters:**
- **Oversupplied station** (too many bikes, start_s > min_s): removing bikes helps → c_s > 0
- **Undersupplied station** (too few bikes, start_s < min_s): adding bikes helps → c_s < 0

---

## 4. The Optimization Model (IP)

### Decision Variables
- `x_{stk} ∈ {0,1}` — is truck k at station s at time t?
- `y_{stk} ≥ 0` — bikes accessible to truck k at station s at time t
- `b_{tk} ≥ 0` — bikes on truck k at time t

### Objective
```
maximize Σ_{s,k} (y_{s,1,k} - y_{s,T,k}) · c_s
```

`(y_{s,1,k} - y_{s,T,k})` is the net bikes removed from station s by truck k. Multiplied by c_s:
- Removing bikes from an oversupplied station (c_s > 0) → positive contribution
- Delivering bikes to an undersupplied station (c_s < 0, so removal is negative) → positive contribution

### Key Constraints
1. **One location per truck**: truck must be at exactly one station at each time step
2. **Movement**: truck can stay or move to an adjacent station each step
3. **Initialization**: trucks start at a depot
4. **Pareto A**: truck can only access bikes at a station if it is physically there
5. **Pareto B**: limits how much the truck can move beyond what's needed
6. **Conservation**: bikes on truck + bikes accessible at stations are conserved
7. **Load when present**: can only load/unload when at the station
8. **Move OR load**: can't do both in the same time step

---

## 5. Data Files

| File | Description |
|---|---|
| `Code/Data/midtown_station_params.csv` | 105 stations, static params (kappa, rates) |
| `Code/Data/nightly_start_s.csv` | ~3,255 rows, one per (station, night) |
| `Code/Data/202501-citi-bike-nyc-stats.parquet` | Raw GBFS station snapshots, January 2025 |
| `Code/Data/202501-citibike-tripdata_1.csv` | Trip data part 1 |
| `Code/Data/202501-citibike-tripdata_2.csv` | Trip data part 2 |
| `Code/Data/202501-citibike-tripdata_3.csv` | Trip data part 3 |

---

## 6. What Has Been Built

### `Code/data_prep.py` — Complete
Reads the raw parquet and trip CSVs, computes all station parameters, and writes the two output CSVs. Key steps:
1. Load station snapshots, compute κ_s
2. Filter to Midtown bounding box (105 stations)
3. For each (station, night): take last snapshot at or before 10pm → `start_s`
4. Load and filter all three trip files to daytime hours
5. Split into weekday and weekend trips
6. Compute μ_s and λ_s separately for each day type
7. Output `midtown_station_params.csv` and `nightly_start_s.csv`

### `Code/udf.py` — Next Step
Will implement the UDF computation: Euler's method on the Kolmogorov ODEs to compute F_s(x_0) for every station and every possible starting count, then derive min_s and c_s.

### `Code/ip.py` — After UDF
Will implement the overnight IP in gurobipy, using the precomputed c_s values as objective coefficients.

---

## 7. Implementation Sequence Going Forward

1. **UDF** (`udf.py`) — Euler's method → F_s curves → min_s → c_s per night
2. **Adjacency graph** — haversine distances between stations, dummy nodes for multi-step travel
3. **IP** (`ip.py`) — gurobipy model with all constraints from the paper
4. **Analysis** — compare dissatisfaction before/after rebalancing, vary K (number of trucks)

---

## 8. Tech Stack
- **Python** with pandas, numpy
- **Gurobi** (gurobipy) — student license — for solving the IP
- **Paper**: Freund et al. (2020) BSS Overnight
