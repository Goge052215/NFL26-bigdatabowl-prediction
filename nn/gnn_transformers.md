# Spatio‑Temporal GNN + Transformers for NFL Trajectory Prediction

This document explains the design in `nn/nn_gnn.py`, why the NFL dataset is not an ordinary serial prediction task, and how the current pipeline combines domain geometry, interaction features, and a spatio‑temporal Transformer. It also proposes feasible performance improvements and their expected impact.

## Why NFL is not an ordinary serial prediction

NFL player trajectory forecasting differs from simple time‑series for several reasons:
- Multi‑agent dynamics: Each player’s motion depends on teammates and opponents (press, separation, mirroring, collisions).
- Strong geometry/physics constraints: Field bounds, momentum, acceleration limits, ball landing point, and coverage offsets.
- Role heterogeneity: Receivers, coverage defenders, QB, OL, RB, LB have distinct priors and behaviors.
- Variable future horizon: The number of output frames differs per play; events like the catch change dynamics mid‑horizon.
- Spatial + temporal dependencies: We must model local spatial neighborhoods and their evolution over time, not just a single stream.

These properties motivate adding geometric baselines and multi‑agent features, then learning corrections with a temporal model.

---

## Pipeline overview (as implemented in `nn_gnn.py`)

- Data: Kaggle CSVs under `Config.DATA_DIR` for weeks 1–18 (`train/input_2023_w*.csv`, `train/output_2023_w*.csv`).
- Sequence building: `prepare_sequences_geometric` constructs per‑player windows with engineered features.
- Targets: Future displacements `(dx, dy)` for up to `MAX_FUTURE_HORIZON` frames, masked per sample because horizons vary.
- Model: `STTransformer` encodes the window and produces horizon‑wise `(dx, dy)` predictions; outputs are integrated with `cumsum`.
- Loss: `TemporalHuber` with time decay emphasizes earlier steps and handles outliers.
- Training: GroupKFold by `game_id`, per‑fold `StandardScaler`, AdamW + LR scheduler, gradient clipping, early stop.
- Inference: Ensemble average across folds; predictions aligned and clamped to field bounds; submission returns only `x, y`.

---

## Input analysis

- Ingestion:
  - Training: Concatenates weekly input/output CSVs; optional DEBUG mode reduces plays for quick iteration.
  - Inference: Polars DataFrames (`test`, `test_input`) converted to Pandas for compatibility.
- Sequence preparation (`prepare_sequences_geometric`):
  - Sorts by `(game_id, play_id, nfl_id, frame_id)` and takes the last `window_size` frames per player.
  - Computes base physics features (velocity, acceleration, momentum, kinetic energy, BMI, etc.).
  - Adds advanced features (opponents, route clusters, neighbor embeddings) and geometric features.
  - Produces a feature matrix of shape `(window_size, feature_dim)` per player.
- Scaling:
  - Per fold, fit `StandardScaler` on training sequences only; transform validation/test with the same scaler to avoid leakage.

## Output analysis

- Targets:
  - For each player, compute future `(x, y)` from output CSVs and convert to displacements `(dx, dy)` relative to the last observed frame.
  - Variable horizon is handled by padding to `MAX_FUTURE_HORIZON` and creating a per‑sample mask of valid steps (`prepare_targets`).
- Model outputs:
  - `ResidualMLPHead` predicts step‑wise deltas for both axes; `torch.cumsum` integrates them to cumulative `(dx, dy)` per timestep.
- Loss:
  - `TemporalHuber(delta=0.5, time_decay=0.03)` applies robust Huber loss with exponential time weighting; masks ensure padded steps don’t contribute.
- Inference formatting:
  - Ensemble predictions are added to last observed player positions and clamped to `[FIELD_X_MIN, FIELD_X_MAX] × [FIELD_Y_MIN, FIELD_Y_MAX]`.
  - The submission strictly returns `x, y` columns ordered exactly like the input `test` template.

---

## Feature engineering

### Geometric baseline and physics
- `compute_geometric_endpoint` and `add_geometric_features` implement a domain‑aware baseline:
  - Receivers converge to ball landing (`ball_land_x/y`).
  - Coverage defenders mirror their targeted receiver via offsets (`mirror_offset_x/y`) when coupled.
  - Other roles default to momentum (velocity × time‑to‑endpoint).
  - All endpoints are clamped to field bounds.
- Derived geometric features include:
  - `geo_vector_x/y`, `geo_distance`, `geo_required_vx/vy`, `geo_required_ax/ay` (clipped), `geo_alignment`.
  - Role‑aware signals like `geo_receiver_urgency` and `geo_defender_coupling`.

### Opponent interaction
- `get_opponent_features`:
  - Nearest opponent distance, closing speed, counts within 3/5 yards.
  - For coverage defenders: tracked receiver velocity, separation, and spatial offsets.

### Route pattern clustering
- `extract_route_patterns`:
  - Features: straightness, max/mean turn, depth/width, speed mean/change over recent frames.
  - Standardized and clustered by KMeans into `N_ROUTE_CLUSTERS` route types.

### GNN‑lite neighbor embeddings
- `compute_neighbor_embeddings`:
  - From the last observed frame, compute ally/opponent weighted relative position/velocity using an exponential kernel with temperature `TAU`.
  - Limit neighbors by top‑`K_NEIGH` and `RADIUS`.
  - Produce counts, min/mean distances, and nearest distances `gnn_d1–d3`.

### Extra derived signals
- Pressure (`1/nearest_opp_dist`), under‑pressure flags, pressure × speed.
- Mirror similarity and alignment features (for coverage roles).

The total feature set targets ~167 features per timestep (≈154 proven features + 13 geometric additions).

---

## Models

### STTransformer (Spatio‑Temporal Transformer)
- Input projection: Linear to `hidden_dim=128`.
- Learnable positional encoding for timesteps; dropout for regularization.
- Temporal encoder: `nn.TransformerEncoder` with `n_layers=2`, `n_heads=4`, GELU FFN of size `hidden_dim × 4`.
- Attention pooling: Learned query attends over the encoded window to produce a context vector.
- Residual MLP Head: Two residual FFN blocks at 256 dims with LayerNorm/GELU map context → `(horizon × 2)` outputs.
- Integration: `cumsum` along time→ cumulative `(dx, dy)`.

### Training & evaluation
- Optimizer: AdamW with weight decay; gradient clipping to `1.0`.
- Scheduler: ReduceLROnPlateau on validation loss; early stopping with `PATIENCE`.
- Cross‑validation: 10‑fold `GroupKFold` by `game_id`; per‑fold scalers and models saved if `SAVE_ARTIFACTS`.
- Metric: `compute_val_rmse` returns RMSE of Euclidean errors over the validation horizon.

---

## GNN

### What’s implemented (feature‑level aggregation)
- A “GNN‑lite” approach: static neighbor summaries from the last observed frame per player.
- Strengths: low compute, robust, captures immediate local topology (who is near, moving how).
- Limitations: no learnable message passing; no explicit cross‑player temporal evolution within the window.

### Potential upgrade (learnable spatio‑temporal graph)
- Construct temporal graphs over all timesteps in the window with K‑NN and radius constraints.
- Apply Graph Attention or a Spatio‑Temporal Graph Transformer to propagate messages across players and time.
- Expected benefit: richer modeling of evolving press/separation and collisions; better defender‑receiver coupling.

---

## Feasibility of performance improvements

Below are practical changes ordered by typical complexity and expected payoff (rough estimates; actual LB gains depend on training stability and metric definition):

1. Residual targets to the geometric baseline (low complexity, low‑to‑moderate payoff)
   - Train the model to predict corrections: `residual = target_dx/dy − geo_dx/dy` (and add back at inference).
   - Benefits: smaller variance, easier learning task grounded in physics/roles.
   - Est. gain: +0.005–0.01.

2. Temporal neighbor embeddings (low‑to‑moderate complexity, moderate payoff)
   - Compute neighbor features per timestep across the window instead of only last frame.
   - Benefits: model sees evolving interactions (closing/opening) over the window.
   - Est. gain: +0.005–0.015.

3. Role‑aware mixture‑of‑experts head (moderate complexity, moderate payoff)
   - Gate the head by role features (e.g., coverage vs receiver vs line).
   - Benefits: respects heterogeneous priors and behaviors across roles.
   - Est. gain: +0.01–0.02.

4. Learnable spatio‑temporal GNN (moderate complexity, moderate‑to‑high payoff)
   - Introduce graph message passing over time with attention and masked adjacency.
   - Benefits: captures multi‑agent couplings directly; complements the Transformer.
   - Est. gain: +0.01–0.03.

5. Mirror coupling improvements (low complexity, low payoff)
   - Robust defender‑receiver pairing across the window; explicit intent vectors for receivers.
   - Est. gain: +0.005–0.01.

6. Loss/time weighting sweeps (low complexity, situational payoff)
   - Tune `time_decay`; consider staged weights for early/mid/late horizons per LB sensitivity.
   - Est. gain: +0.002–0.01.

7. Window length tuning (low complexity, small payoff)
   - Try `window_size` in 12–16 for more pre‑snap context, or shorter to reduce noise.
   - Est. gain: +0.002–0.008.

8. Ensembles and seeds (low complexity, additive payoff)
   - Ensemble across seeds and small architecture variants (GRU/LSTM/Transformer/ST‑GNN, residual vs absolute targets).
   - Est. gain: +0.01–0.02.

---

## Implementation sketches

### Residual target training (concept)

- Compute geometric displacements to endpoint and distribute across horizon (simple linear schedule or using `geo_required_vx/vy`).
- Train on residuals:

```python
# During sequence/target prep (conceptual)
geo_dx_seq = make_geo_dx_sequence(last_state, horizon, geo_features)
geo_dy_seq = make_geo_dy_sequence(last_state, horizon, geo_features)
res_dx = dx_true[:L] - geo_dx_seq[:L]
res_dy = dy_true[:L] - geo_dy_seq[:L]
# Train on residuals; at inference: pred_res + geo_seq → final dx/dy
```

Notes:
- Keep masks aligned; ensure residuals use the same integration convention as the model (stepwise deltas vs cumulative).
- Start with a simple linear schedule for geometric deltas if endpoint timing is uncertain; refine with role‑specific schedules later.

### Temporal neighbor embeddings (concept)

For each timestep in the window:
- Build K‑NN within `RADIUS` using player positions.
- Compute ally/opponent weighted stats (relative position/velocity, counts, distances) like `compute_neighbor_embeddings`.
- Concatenate per‑timestep neighbor features to the feature vector before the Transformer.

---

## Risks and considerations

- Overfitting late horizon: guard with `time_decay`, regularization, and early stopping.
- Scaling leakage: always fit scalers on training splits only.
- Graph compute cost: cap neighbors (`K_NEIGH`) and use distance kernels (`TAU`) to stabilize.
- Role imbalance: role‑aware heads need regularization and sufficient data per role.
- Submission ordering: ensure alignment strictly follows the input `test` template.

---

## Validation & metrics

- `compute_val_rmse` computes per‑timestep Euclidean errors and aggregates RMSE; useful for sanity checks.
- Consider reporting by role or distance bins (near/under pressure) to diagnose where the model helps most.

---

## Next steps

- Add residual target training (least invasive change; pairs well with existing geometric features).
- Extend neighbor embeddings temporally and/or add a lightweight graph attention block.
- Prototype role‑aware mixture‑of‑experts in the output head.
- Run controlled ablations and report fold‑wise RMSE deltas.

With these steps, the current geometry‑anchored Transformer can better reflect the non‑serial, multi‑agent nature of NFL trajectories and should yield incremental leaderboard improvements while keeping training stable and interpretable.

---

## Dataset properties analysis and pipeline rationale

Below is a deeper analysis of the NFL prediction dataset’s properties and how they inform the design choices in this pipeline.

### 1) Data schema and entities
- Per‑frame tables with keys: `game_id`, `play_id`, `nfl_id`, `frame_id`.
- Player kinematics: `x, y, s (speed), dir (direction in degrees), a (acceleration)`.
- Player attributes: `player_height`, `player_weight`, and role/sides like `player_role`, `player_side`.
- Output frames: future coordinates for the same `(game_id, play_id, nfl_id)` indexed by `frame_id`; horizon length varies per play and may be available via `num_frames_output`.
- Implication: Multi‑index grouping and variable‑length sequence handling are mandatory.

### 2) Spatio‑temporal resolution and limits
- The dataset runs at ~10 Hz (the code uses `num_frames_output/10.0` → seconds), so positions change meaningfully across ~10 consecutive frames.
- Field bounds are fixed: `[0, 120] × [0, 53.3]` yards; physical motion is limited by realistic speed/acceleration.
- Implication: Clipping predictions to field bounds prevents physically impossible outputs; short windows (e.g., 10 frames) are sufficient to capture immediate momentum and intent pre‑catch.

### 3) Role heterogeneity and regime changes
- Roles differ sharply: receivers run routes toward ball landing, defenders maintain mirrors/offsets, OL/RB/LB have blocking/following behaviors.
- Plays have phases (pre‑snap, release, ball in flight, catch/contested): dynamics shift mid‑horizon.
- Implication: A single stationary model is suboptimal; role‑aware features and baselines (ball convergence for receivers, mirroring for defenders) stabilize learning; time‑weighted loss reduces late‑horizon volatility impact.

### 4) Multi‑agent coupling and locality
- A player’s motion is coupled with neighbors (teammates and opponents) via spacing, leverage, and closing speed.
- Local neighborhoods dominate short‑term changes; faraway players matter less.
- Implication: Neighbor embeddings with distance‑weighted aggregation (GNN‑lite) are effective and inexpensive; extending them temporally would capture evolving interactions better.

### 5) Noise, missingness, and non‑Gaussian errors
- Sensor noise in `dir/s/a` and occasional missing entries can cause outliers.
- Late‑horizon labels are more uncertain due to branching trajectories (multi‑modality) and play‑specific events.
- Implication: Robust losses (Huber) and time decay reduce sensitivity to outliers and uncertain future steps; masks prevent padded steps from corrupting loss.

### 6) Variable future horizon and test template alignment
- Different plays have different numbers of output frames; the test template supplies exact `frame_id`s to predict.
- Implication: Padding + masks are essential in training; during inference, predictions must be re‑indexed to the provided template strictly and returned as `x, y` only.

### 7) Distribution shift and leakage risks
- Plays within the same game share conditions (personnel, field, weather), so splits by random rows risk leakage.
- Implication: Use `GroupKFold` by `game_id` to reduce leakage and approximate realistic generalization across games.

### 8) Geometry‑anchored baseline aids generalization
- Receivers → ball landing; defenders → mirrored offsets; others → momentum.
- Implication: Training on corrections to this baseline (residual targets) simplifies the task and improves robustness, especially under role heterogeneity and phase changes.

### How these properties justify pipeline choices
- Short temporal window + Transformer: captures recent intent/accel while remaining compute‑efficient at 10 Hz.
- Learnable positional encoding: preserves order information without relying on absolute timestamps.
- Attention pooling: compresses the window into a context vector robust to noisy frames.
- Residual MLP head: provides capacity and stability for mapping context to horizon outputs.
- Huber + time weighting + masks: address noise, multi‑modality, and variable horizon.
- Neighbor embeddings and route clustering: encode local coupling and typical intent patterns.
- Geometry features: anchor learning with strong priors and reduce variance.
- GroupKFold by game: mitigate leakage and improve generalization estimates.

### Properties‑anchored improvement pointers
- Temporal neighbor embeddings or ST‑GNN: target evolving local coupling over the window.
- Residual target training: align learning with the geometric baseline to counter role/regime heterogeneity.
- Role‑aware MoE head: respect heterogeneous priors to reduce bias and variance.
- Loss/weight sweeps: adjust time decay to match uncertainty profiles across the horizon.