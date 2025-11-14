## Dataset Portrait
- Schema: `input_*` contains per-frame kinematics (`x`,`y`,`s`,`a`,`dir`,`o`), player metadata, `player_to_predict`, and pass landing (`ball_land_x`,`ball_land_y`); `output_*` contains future target coordinates (`x`,`y`) per targeted player and frame.
- Targets: Sequence length varies by play; horizon capped to `55` frames in code. Ids group by `(game_id, play_id, nfl_id)`.
- Direction normalization: Inputs are unified to left; outputs re-inverted using play-direction map.
- Notable signals: Kinematics, ball geometry (distance/angle to landing), player role/side, short-term trends, neighbor pressure/density.

## Current Model & Pipeline
- Feature builder: Modular groups with high coverage: `target_alignment`, `lag`, `motion_change`, `field_position`, `distance_rate`, `geometric`, `neighbor_gnn`, `time`, `passer`, `curvature`, `route`, `receiver` (nn/nn-gnn-localtest.py:314).
- Play-level sequences: `[P, T, D]` with player mask; time-varying neighbor pressure/density appended (nn/nn-gnn-localtest.py:1987, 2094–2162).
- ST-Transformer: Player+time positional encodings; vanilla TransformerEncoder over flattened `P×T` tokens; anchored pooling on targeted player's last time token; Residual MLP head with cumsum deltas (nn/nn-gnn-localtest.py:1360–1474).
- Loss: Temporal Huber with horizon decay (nn/nn-gnn-localtest.py:1278–1315). Optimizer AdamW + ReduceLROnPlateau; grad clip; early stop (nn/nn-gnn-localtest.py:1526–1604).
- CV: GroupKFold by game+play; multi-seed support (nn/nn-gnn-localtest.py:2234–2259).

## Gaps vs Roadmap
- Ball-centric interactions: No explicit shared ball token; ball features are per-player scalars in base features.
- Hierarchical player-axis attention: Current encoder is flat over `P×T`; no explicit cross-attention between target receiver and defenders.
- Residual ensemble: Not implemented; single model governs bias.
- Physics-informed post-processing: Basic cumsum only; no smoothing or speed/acc clamps.
- Feature compactness: Many groups active; role features disabled in `Config.FEATURE_GROUPS`; lag window includes long lags; multi-window EMA/STD disabled but available.

## Upgrade Recommendations
- Features (compact, high-signal):
  - Enable `role` group; keep `target_alignment`, `time`, `distance_rate`, `route` (last-5), `receiver`, `passer`, and short `lag` (1–3) while dropping `lag10` and heavy `multi_window` in first pass to reduce noise.
  - Keep `neighbor_gnn` but use K=3 and RADIUS≈25–30; retain time-varying pressure/density.
  - Verify curvature usefulness; if unstable, keep `curvature_abs` only.
- Ball-centric multi-player modeling:
  - Add a shared ball token per frame: features such as `ball_direction_x/y`, `distance_to_ball`, `angle_diff`, and global `pass_direction`; append as player `P+1` with `player_mask=1` to let attention route through ball.
  - Add receiver-to-defender cross-attention block: compute attention with queries from target receiver tokens over defender tokens at last K frames; concatenate attended context with anchored pool before head.
- ST-Transformer upgrades:
  - Axial attention: two-stage blocks — player-axis attention per time slice, then temporal attention over the target; retain positional encodings.
  - Multi-query pooling: set `n_queries=2–4` and learn separate query vectors for target last-time, defender-context, and ball-token context; concatenate into head.
  - Head: keep ResidualMLP; consider increasing `MLP_HIDDEN_DIM` to 384 and `N_RES_BLOCKS=3`.
  - Loss: tune `TemporalHuber` with `delta≈0.35–0.45` and `time_decay≈0.05–0.08`; optionally weight first 10 frames higher to stabilize close-horizon.
  - Training: add cosine schedule with warmup (10% steps), label smoothing on deltas, and stochastic depth 0.05 for encoder layers to improve generalization.
- Residual tree ensemble:
  - Train small CatBoost/XGB/LGB on OOF residuals using compact features (last-frame kinematics, nearest-opponent pressure/density, role flags); predict residual and add to NN outputs. Keep blending simple (NN + residual, α≈0.7–0.85).
- Physics-informed post-processing:
  - Temporal smoothing on deltas with EMA (α≈0.3) and Savitzky–Golay alternative; clamp implied speed ≤9 yards/s and acceleration ≤5 yards/s²; add role-specific regularization for receivers near sideline.

## Milestones & Expected Gains
- M1 Features compactness and role enablement: −0.010 to −0.015 RMSE.
- M2 ST-Transformer axial attention + ball token + multi-query pooling: −0.015 to −0.025 RMSE.
- M3 Residual ensemble and smoothing: −0.010 to −0.015 RMSE.
- M4 Minimal blend tuning and horizon-weight loss: −0.005 to −0.010 RMSE.
- Combined target: from ~0.562 → ≈0.50; stretch 0.48–0.49 with tuning.

## Document Update (to insert)
- Add "Roadmap to ≤0.50 RMSE" at the top of `nn/gnn_transformers.md` with bullets matching the above: goal, compact features, ball-centric interactions, ST-Transformer upgrades, residual ensemble, physics-informed post-processing, milestones, and an expected gains table.

## File Touchpoints
- ST-Transformer and loss: nn/nn-gnn-localtest.py:1278, 1339, 1360–1474.
- Feature groups and neighbor features: nn/nn-gnn-localtest.py:314–846, 646–846, 2094–2162.
- Training loop and CV: nn/nn-gnn-localtest.py:1480–1604, 1648–1735, 2195–2263.

## Next Step
- Implement the upgrades in `nn/nn-gnn-localtest.py`, run CV to verify gains, then append the roadmap section to `nn/gnn_transformers.md` and summarize outcomes in the Bronze Path document. 