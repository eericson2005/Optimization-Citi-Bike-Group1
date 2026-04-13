"""
Data preparation pipeline for overnight Citi Bike rebalancing optimization.

Outputs:
  1. Data/midtown_station_params.csv  — one row per station (static parameters)
       Columns: station_name, latitude, longitude, kappa,
                mu_s_weekday, lambda_s_weekday, mu_s_weekend, lambda_s_weekend

  2. Data/nightly_start_s.csv         — one row per (station, night)
       Columns: date, station_name, start_s

Scope:
  - Geographic: Midtown Manhattan (~34th–59th St)
  - Temporal:   January 2025
  - Daytime:    7am–10pm (for rate estimation)
  - Overnight:  10pm–7am (start_s taken from first snapshot at or after 10pm each night)
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATS_PATH = "Data/202501-citi-bike-nyc-stats.parquet"
TRIPS_PATHS = [
    "Data/202501-citibike-tripdata_1.csv",
    "Data/202501-citibike-tripdata_2.csv",
    "Data/202501-citibike-tripdata_3.csv",
]
PARAMS_OUTPUT_PATH = "Data/midtown_station_params.csv"
NIGHTLY_OUTPUT_PATH = "Data/nightly_start_s.csv"

# Midtown Manhattan bounding box (~34th St to ~59th St).
# Western limit -74.012 captures Hudson Yards / far-west stations.
# Eastern limit -73.960 captures 1st Ave while excluding Roosevelt Island and LIC.
LAT_MIN, LAT_MAX = 40.748, 40.765
LON_MIN, LON_MAX = -74.012, -73.960

# Daytime window used for rate estimation (inclusive start, exclusive end).
# Rates represent demand during the operating day that follows an overnight
# rebalancing run — trucks set starting conditions, then users ride during the day.
DAY_START_HOUR = 7  # 7am
DAY_END_HOUR = 22  # 10pm  (trips with started_at hour in [7, 21])

DAYTIME_MINUTES_PER_DAY = (DAY_END_HOUR - DAY_START_HOUR) * 60  # 900 minutes


# ---------------------------------------------------------------------------
# Step 1: Load station stats, extract capacity and geometry
# ---------------------------------------------------------------------------

print("Loading station stats parquet...")
stats = pd.read_parquet(STATS_PATH)
stats["timestamp"] = pd.to_datetime(stats["timestamp"])
stats["hour"] = stats["timestamp"].dt.hour

# Capacity (kappa): use max observed (bikes + free) across all January snapshots.
# We do NOT use the 'slots' field from the extra JSON because ~87% of records show
# bikes + free != slots, caused by disabled docks that reduce usable capacity.
# Taking the max of (bikes + free) over all snapshots gives the highest observed
# active capacity — a reliable upper bound on the true dock count that naturally
# excludes disabled docks.
stats["active_cap"] = stats["bikes"] + stats["free"]
stats["kappa"] = stats.groupby("name")["active_cap"].transform("max")

# Filter to Midtown bounding box
midtown_stats = stats[
    (stats["latitude"] >= LAT_MIN)
    & (stats["latitude"] <= LAT_MAX)
    & (stats["longitude"] >= LON_MIN)
    & (stats["longitude"] <= LON_MAX)
].copy()

print(f"  Midtown stations found: {midtown_stats['name'].nunique()}")

# ---------------------------------------------------------------------------
# Step 2: Build station-level geometry and capacity table
# ---------------------------------------------------------------------------

# Creates a lookup table for midtown stations that includes location and capacity
station_geo = (
    midtown_stats.groupby("name")
    .agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        kappa=("kappa", "max"),
    )
    .reset_index()
)

# ---------------------------------------------------------------------------
# Step 3: Compute nightly start_s — bike count at 10pm each night
# ---------------------------------------------------------------------------
# Trucks rebalance every night, so start_s is NOT a static monthly average.
# Instead, for each (station, night) we take the last GBFS snapshot at or
# before 10pm — this is the actual starting condition that night's truck routes
# will operate on.
#
# We use "last snapshot at or before 10pm" rather than "first snapshot in the
# 10pm hour" because the parquet is polled every ~3 minutes and gaps can span
# an entire hour. Filtering to hour==22 would silently drop any (station, night)
# pair where the archiver happened to miss the 10pm window. Taking the last
# snapshot at or before 10:00pm guarantees coverage for every station/night
# that has any data that day.
#
# We save one row per (station, date) so the IP solver can load the correct
# starting conditions for whichever night is being optimized.

import datetime

# Extract just the date information from the timestamp and store it as a new column
midtown_stats["date"] = midtown_stats["timestamp"].dt.date
cutoff_time = datetime.time(22, 0, 0)  # Creates a cutoff time object = 10:00 p.m.

# Filters midtown_stats for info before or at 10:00 p.m. and saves as a copy
at_or_before_10pm = midtown_stats[
    midtown_stats["timestamp"].dt.time <= cutoff_time
].copy()

# Sort ascending so .last() gives the snapshot closest to (but not after) 10:00pm
at_or_before_10pm_sorted = at_or_before_10pm.sort_values("timestamp")
# Creates the lookup table for start_s for each station, for each night
first_snap = (
    # Must include date in groupby otherwise we would grab only last days of month
    at_or_before_10pm_sorted.groupby(["name", "date"])
    .last()
    .reset_index()[["name", "date", "bikes"]]
    .rename(columns={"name": "station_name", "bikes": "start_s"})
)

# Drop NA from results
first_snap.dropna(subset=["start_s"], inplace=True)
print(
    f"  Stations with at least one pre-10pm snapshot: {first_snap['station_name'].nunique()}"
)
print(f"  Total (station, night) rows: {len(first_snap)}")

# ---------------------------------------------------------------------------
# Step 4: Load trip data and filter to daytime hours
# ---------------------------------------------------------------------------
# Here we grab the trip data and filter it for just the daytime hours so that we
# can estimate mu and lambda for each station

print("\nLoading trip data CSVs...")
# Load in trip data and join everything together
trips = pd.concat(
    [
        pd.read_csv(p, parse_dates=["started_at", "ended_at"], low_memory=False)
        for p in TRIPS_PATHS
    ],
    ignore_index=True,
)
# Extract just the trip start hour and make it it's own column
trips["start_hour"] = trips["started_at"].dt.hour

# Filter trips to just included daytime trips
daytime_trips = trips[
    (trips["start_hour"] >= DAY_START_HOUR) & (trips["start_hour"] < DAY_END_HOUR)
].copy()

print(f"  Total trips loaded:   {len(trips):,}")
print(f"  Daytime trips (7-22): {len(daytime_trips):,}")

# ---------------------------------------------------------------------------
# Step 5: Split daytime trips into weekday and weekend
# ---------------------------------------------------------------------------
# Demand at Citi Bike stations differs substantially between weekdays and
# weekends — commuter-heavy stations near offices spike Mon–Fri, while
# leisure-oriented stations are busier on weekends.
#
# Blending weekday and weekend trips into a single rate would produce an
# average that accurately represents neither day type. Instead, we compute
# separate rates so the IP solver can use the rate that matches the coming
# day (e.g., trucks running Sunday night use weekend rates for Monday).
#
# dayofweek: Monday=0 ... Sunday=6; weekend = Saturday(5) or Sunday(6)

# Extract date information for weekend vs weekday distinction
daytime_trips["date"] = daytime_trips["started_at"].dt.date
# Create is_weekend flag
daytime_trips["is_weekend"] = daytime_trips["started_at"].dt.dayofweek >= 5

# Create two views of daytime trips, one for weekdays and one for weekends
weekday_trips = daytime_trips[~daytime_trips["is_weekend"]]
weekend_trips = daytime_trips[daytime_trips["is_weekend"]]

print(f"\n  Weekday daytime trips: {len(weekday_trips):,}")
print(f"  Weekend daytime trips: {len(weekend_trips):,}")

# ---------------------------------------------------------------------------
# Step 6: Compute mu_s and lambda_s for weekday and weekend
# ---------------------------------------------------------------------------
# For each day type:
#   1. Count trips per (station, date) — this gives the raw daily activity.
#   2. Average those daily counts across all observed dates of that type —
#      this gives the expected number of trips on a typical day of that type.
#   3. Divide by DAYTIME_MINUTES_PER_DAY (900) to convert to trips/minute.
#
# Per-day averaging (rather than summing the month and dividing by total
# minutes) correctly handles unequal numbers of weekday vs. weekend days
# and is robust to any days with missing data.
#
# Cross-boundary trips are included intentionally: mu_s and lambda_s measure
# activity AT station s, regardless of where the other end of the trip is.
# A bike departing a Midtown station to an outside destination still depletes
# one dock; a bike arriving at a Midtown station from outside still fills one.


def compute_rates(trip_df, dep_col, arr_col, suffix):
    """Return (mu_series, lambda_series) named with given suffix."""
    # Here's how to read this code:
    # First, group by station and by date.
    # Then, output the size i.e. number of events for a station for a given day
    # Repeat for all stations, for all days.
    # Then, take all results for a given station and find their mean
    # Finally, divide these monthly averages by the total minuetes in a day to get a rate
    daily_dep = trip_df.groupby([dep_col, "date"]).size().groupby(dep_col).mean()
    daily_arr = trip_df.groupby([arr_col, "date"]).size().groupby(arr_col).mean()
    mu = (daily_dep / DAYTIME_MINUTES_PER_DAY).rename(f"mu_s_{suffix}")
    lam = (daily_arr / DAYTIME_MINUTES_PER_DAY).rename(f"lambda_s_{suffix}")
    return mu, lam


# Compute rates for weekday and weekend respectively
mu_weekday, lambda_weekday = compute_rates(
    weekday_trips, "start_station_name", "end_station_name", "weekday"
)
mu_weekend, lambda_weekend = compute_rates(
    weekend_trips, "start_station_name", "end_station_name", "weekend"
)

# ---------------------------------------------------------------------------
# Step 7: Join everything into the station params table
# ---------------------------------------------------------------------------
# The station_geo table already holds station location and capacity.
# Now, the monthly rates get added in.

params = (
    station_geo.set_index("name")
    .join(mu_weekday, how="left")
    .join(lambda_weekday, how="left")
    .join(mu_weekend, how="left")
    .join(lambda_weekend, how="left")
    .reset_index()  # Changes station name back to column from an index
    .rename(columns={"name": "station_name"})
)

# Stations with zero observed trips of a given type get rate = 0.
# This is correct: no trips observed → no demand estimated.
for col in ["mu_s_weekday", "lambda_s_weekday", "mu_s_weekend", "lambda_s_weekend"]:
    params[col] = params[col].fillna(0.0).round(6)

# Reset row numbers from the joins
params = params.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 8: Summary and output
# ---------------------------------------------------------------------------

print("\n=== Midtown Station Parameters ===")
print(f"  Stations in output: {len(params)}")
print(
    f"  Capacity (kappa)        — mean: {params['kappa'].mean():.1f}, "
    f"min: {params['kappa'].min()}, max: {params['kappa'].max()}"
)
print(
    f"  mu_s weekday (pick/min) — mean: {params['mu_s_weekday'].mean():.4f}, "
    f"max: {params['mu_s_weekday'].max():.4f}"
)
print(
    f"  mu_s weekend (pick/min) — mean: {params['mu_s_weekend'].mean():.4f}, "
    f"max: {params['mu_s_weekend'].max():.4f}"
)
print(
    f"  lambda_s weekday        — mean: {params['lambda_s_weekday'].mean():.4f}, "
    f"max: {params['lambda_s_weekday'].max():.4f}"
)
print(
    f"  lambda_s weekend        — mean: {params['lambda_s_weekend'].mean():.4f}, "
    f"max: {params['lambda_s_weekend'].max():.4f}"
)

for col in ["mu_s_weekday", "lambda_s_weekday", "mu_s_weekend", "lambda_s_weekend"]:
    print(f"  Stations with {col} = 0: {(params[col] == 0).sum()}")

print("\nSample station params:")
print(
    params[
        [
            "station_name",
            "kappa",
            "mu_s_weekday",
            "lambda_s_weekday",
            "mu_s_weekend",
            "lambda_s_weekend",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\n=== Nightly start_s ===")
print(f"  Total (station, night) rows: {len(first_snap)}")
print(f"  Nights covered: {first_snap['date'].nunique()}")
print("\nSample nightly start_s:")
print(first_snap.head(10).to_string(index=False))

params.to_csv(PARAMS_OUTPUT_PATH, index=False)
print(f"\nSaved station params to {PARAMS_OUTPUT_PATH}")

first_snap.to_csv(NIGHTLY_OUTPUT_PATH, index=False)
print(f"Saved nightly start_s to  {NIGHTLY_OUTPUT_PATH}")
