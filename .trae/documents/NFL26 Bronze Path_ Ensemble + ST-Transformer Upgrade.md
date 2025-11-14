## Target Document

* Append a new section to `nn/gnn_transformers.md`: "Roadmap to ≤0.50 RMSE".

## Section Contents

* Goal and current baseline (\~0.562 RMSE) and target (≤0.50; aspirational 0.48–0.49).

* Compact, high-signal feature set (avoid noisy stacks): kinematics, ball geometry, role/mirror/route, short lags/EMAs, lean GNN-lite.

* Ball-centric multi-player interactions: shared ball token, receiver-to-defender cross-attention, nearest-opponent pressure/density.

* ST-Transformer upgrades: hierarchical player-axis attention, Residual MLP head, temporal Huber with horizon decay, training stability.

* Lightweight residual tree ensemble for bias correction only (CatBoost + XGB/LGB residuals) and minimal blending.

* Physics-informed post-processing: smoothing, speed/acc constraints, role-specific regularization.

* Milestones: M1 features, M2 upgraded STT ≤0.53, M3 residual+smoothing ≤0.51, M4 final blend ≈0.50.

* Expected gains table: per module incremental RMSE reductions to reach \~0.50.

## Style & Placement

* Use concise bullet lists and keep emphasis on essentials, avoiding exhaustive feature lists.

* Place the roadmap near the top of the document, before detailed architecture notes, with internal anchors.

## Next Step

* On approval, insert the section and commit the doc update.

## Current Progress

1. XGB + CatBoost + LightGBM = 0.74 RMSE
2. Better feature Engineer + CatBoost and GNN = 0.63 RMSE
3. Simple Neural Network with 102 features = 0.61 RMSE
4. Model 3 but higher LR:4e-4, Conv1D, 114 features=0.598 RMSE
5. Post Competition-Paused: Geometric GNN + Spatio-Transformer yields a 0.580 RMSE
6. nn-gnn-localtest.py has a best 0.562 RMSE with ST-Transformer. (Current version)

* Kaggle changed to API submission, which severly affected our codebase. Now we restart our work based on the nn/nn\_gnn.py. When we submit our work, we need to ensure that the generation of parquet files models, otherwise the API will fail when the model cant be loaded.
  I hope the record above helps you to understand more about strategy selection.

## M1: Features Compactness and Role Enablement

* Changes:

  * Enabled role features and retained high-signal groups: target\_alignment, lag (1–3), distance\_rate, time, passer, route (last-5), receiver, neighbor\_gnn.

  * Reduced neighbor parameters to K\_NEIGH=3 and RADIUS=28.0.

  * Restricted curvature to curvature\_abs only.

* Rationale:

  * Compact features reduce noise and overfitting while keeping ball-centric and short-horizon dynamics; role-aware signals improve contextual modeling.

* Debug CV Result:

  * Play-level ST-Transformer, 5-fold GroupKFold; mean RMSE 0.6773, best fold 0.5719 (debug-sized dataset).

  * Expect ≈0.010–0.015 RMSE improvement on full data after tuning.

* Next:

  * M2: add ball token, axial player-axis attention, and multi-query pooling.

## M2: ST-Transformer Upgrades — Implementation Update

- Scope: Implemented M2 in `nn/nn-gnn-localtest.py` with ball-centric multi-player modeling and hierarchical attention.
- Data changes:
  - Adds a shared ball token to the player axis per play, using `ball_land_x/ball_land_y` across the input window.
  - Extends `player_mask` to include the ball token; preserves existing `tv_*` neighbor features untouched.
  - Code reference: `nn/nn-gnn-localtest.py:2024` (`prepare_sequences_play_level`).
- Model changes:
  - Axial attention: temporal encoder per player → player-axis encoder at the last time step with key padding masks.
  - Multi-query pooling from player-axis output: target receiver (idx 0), ball token (last idx), nearest player (idx 1).
  - Code reference: `nn/nn-gnn-localtest.py:1354` (`class STTransformer`).
- Config switches:
  - `N_QUERYS=3`, `ADD_BALL_TOKEN=True`, `USE_AXIAL_ATTENTION=True`.
  - Code reference: `nn/nn-gnn-localtest.py:26` (`class Config`).
- Verification:
  - Debug training runs across 5 GroupKFold folds complete successfully; artifacts and metrics written.
  - Metrics path: `output/default/cv_metrics.json`; this debug run is for shape/stability, not indicative of full-data RMSE.
- Expected impact:
  - With full-data training and tuning, anticipate ~0.01–0.02 RMSE reduction relative to the 0.562 baseline.

## Updated Roadmap

- M2 (done locally): ball token, axial player-axis attention, multi-query pooling.
- M3 (next): Residual ensemble (CatBoost + XGB/LGB) for bias correction only; physics-informed smoothing (speed/acc constraints) applied to STT outputs.
- M4: Minimal blending to target ≤0.50 RMSE (aspirational 0.48–0.49).
