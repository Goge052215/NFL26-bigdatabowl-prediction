import os
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Load the data
print("Loading data...")
base_dirs = [
    Path(os.getenv("NFL26_SUPP_DATA_DIR", "")),
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent / "supplement",
    Path(__file__).resolve().parent.parent / "data",
    Path("/data"),
]
candidate_dirs = [d for d in base_dirs if str(d)]
player_play_df = None
frame_df = None
for d in candidate_dirs:
    pp = d / "sumer_coverages_player_play.parquet"
    ff = d / "sumer_coverages_frame.parquet"
    if pp.exists() and ff.exists():
        player_play_df = pl.read_parquet(str(pp))
        frame_df = pl.read_parquet(str(ff))
        break
if player_play_df is None or frame_df is None:
    raise FileNotFoundError(
        "Missing supplementary parquet files: place them in 'data/' or set NFL26_SUPP_DATA_DIR"
    )

print(f"Player-play data: {len(player_play_df):,} rows")
print(f"Frame data: {len(frame_df):,} rows")

print(player_play_df.columns)
print(player_play_df['alignment'].unique().to_list())

# Get coverage scheme columns
scheme_cols = [
    col for col in frame_df.columns 
    if col.startswith("coverage_scheme__")
]
print(
    f"\nCoverage schemes: {[col.replace('coverage_scheme__', '') for col in scheme_cols]}"
)

### Disguise Developments¶
# One of the great things about having frame level coverage data is that 
# we can see how coverages (looks) evolve as the play progresses, 
# and what the trends of 2023 were.

# ============================================================================
# ANALYSIS: First Frame vs Last Frame Coverage Evolution
# ============================================================================

print("\n" + "="*80)
print("COVERAGE SCHEME EVOLUTION: FIRST FRAME VS LAST FRAME")
print("="*80)

# For each play, get first and last frame
play_evolution = (
    frame_df
    .sort(["play_id", "frame_id"])
    .group_by("play_id")
    .agg([
        pl.first("frame_id").alias("first_frame"),
        pl.last("frame_id").alias("last_frame"),
        *[pl.first(col).alias(f"{col}_first") for col in scheme_cols],
        *[pl.last(col).alias(f"{col}_last") for col in scheme_cols]
    ])
)

# Get the dominant scheme for first and last frames
def get_dominant_scheme(row, suffix):
    """Get the scheme with highest probability."""
    scheme_probs = {
        col.replace(f"coverage_scheme__", "").replace(f"_{suffix}", ""): row[col]
        for col in row.keys() if col.startswith("coverage_scheme__") and col.endswith(f"_{suffix}")
        and row[col] is not None
    }
    if scheme_probs:
        return max(scheme_probs.items(), key=lambda x: x[1])[0]
    return None

# Convert to pandas for easier manipulation
evo_pd = play_evolution.to_pandas()
evo_pd['first_scheme'] = evo_pd.apply(lambda r: get_dominant_scheme(r, 'first'), axis=1)
evo_pd['last_scheme'] = evo_pd.apply(lambda r: get_dominant_scheme(r, 'last'), axis=1)
evo_pd['scheme_changed'] = evo_pd['first_scheme'] != evo_pd['last_scheme']

# Summary statistics
print(f"\nTotal plays analyzed: {len(evo_pd):,}")
print(f"Plays with scheme change: {evo_pd['scheme_changed'].sum():,} ({evo_pd['scheme_changed'].sum()/len(evo_pd)*100:.1f}%)")

# Distribution of schemes at first frame
print("\n" + "-"*80)
print("COVERAGE SCHEME DISTRIBUTION AT FIRST FRAME (SNAP)")
print("-"*80)
first_scheme_dist = evo_pd['first_scheme'].value_counts()
print(first_scheme_dist.to_string())
print(f"\nMost common at snap: {first_scheme_dist.index[0]} ({first_scheme_dist.iloc[0]/len(evo_pd)*100:.1f}%)")

# Distribution of schemes at last frame
print("\n" + "-"*80)
print("COVERAGE SCHEME DISTRIBUTION AT LAST FRAME (END OF PLAY)")
print("-"*80)
last_scheme_dist = evo_pd['last_scheme'].value_counts()
print(last_scheme_dist.to_string())
print(f"\nMost common at end: {last_scheme_dist.index[0]} ({last_scheme_dist.iloc[0]/len(evo_pd)*100:.1f}%)")

# Changes in scheme popularity
print("\n" + "-"*80)
print("EFFECTIVE DISGUISE TRENDS (First Frame → Last Frame)")
print("-"*80)
for scheme in sorted(set(first_scheme_dist.index) | set(last_scheme_dist.index)):
    first_pct = (first_scheme_dist.get(scheme, 0) / len(evo_pd) * 100)
    last_pct = (last_scheme_dist.get(scheme, 0) / len(evo_pd) * 100)
    change = last_pct - first_pct
    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
    print(f"  {scheme:15s} {first_pct:5.1f}% → {last_pct:5.1f}% ({arrow} {abs(change):4.1f}%)")

