## Overview

Apply M2 from the roadmap to the local ST-Transformer pipeline in `nn/nn-gnn-localtest.py`: add a shared ball token, switch to axial (time-then-player) attention, and use multi‑query pooling for the output head.

## Goals

* Add a ball-centric token to the player axis for every play.

* Use hierarchical axial attention: temporal per player, then player-axis at the last time step.

* Pool multiple queries (target + ball + nearest players) before the head.

## Where To Change

* Data: `prepare_sequences_play_level` in `nn/nn-gnn-localtest.py:1979–2184`.

* Model: `STTransformer` in `nn/nn-gnn-localtest.py:1352–1469`.

* Config: `Config` in `nn/nn-gnn-localtest.py:24–104`.

## Step 1 — Ball Token (Data)

* Extend the per-play tensor by 1 row to hold a shared ball token across time: `play_tensor -> [P+1, T, D]`.

* Populate ball token features per frame:

  * Set `x=y=(ball_land_x, ball_land_y)` across the window; keep other features zero except fields that are trivially definable (e.g., `frame_id`).

  * Mark the ball token as valid in `player_mask`.

* Leave ordering as: index 0 = targeted receiver, 1..(P−1) = nearest players, last index = ball token.

* Keep scaler logic unchanged (it flattens `[-1, D]`).

## Step 2 — Axial Attention (Model)

* Introduce a two-stage encoder inside `STTransformer`:

  * Temporal encoder over time for each player: reshape to `[B*P, T, H]`, add `pos_time`, encode, then reshape back to `[B, P, T, H]`.

  * Player-axis encoder at last time step: slice `h[:, :, T-1, :] -> [B, P, H]`, add `pos_player` (sized to `P` or `P+1` if ball token), and encode with a second `TransformerEncoder`.

* Maintain `player_mask` to generate a key padding mask for the player-axis encoder. If ball token exists, extend the mask to include it.

## Step 3 — Multi-Query Pooling

* Set `Config.N_QUERYS = 3`.

* Build query representations from the player-axis output at the last time step:

  * q0: targeted receiver (index 0).

  * q1: ball token (last index).

  * q2: nearest other player (index 1). If the ball token is absent, use indices `[0..N_QUERYS-1]`.

* Concatenate queries (`[B, N_QUERYS*H]`) and feed the `ResidualMLP` head to predict `[B, HORIZON, 2]` deltas, followed by temporal cumulative sum.

## Step 4 — Config Switches

* Add flags for clarity and ablation:

  * `ADD_BALL_TOKEN = True`

  * `USE_AXIAL_ATTENTION = True`

  * `N_QUERYS = 3`

* Keep existing sizes: `WINDOW_SIZE=10`, `MAX_PLAYER=9`, `HIDDEN_DIM=128`, `N_HEADS=4`, `N_LAYERS=2`.

## Step 5 — Integration Details

* Inputs: The play-level builder already contains `is_defense` and `tv_*` features. Ordering by nearest distance is done in `_order_players_for_play` (`nn/nn-gnn-localtest.py:1957–1976`), so index 1 is “nearest overall” and is an acceptable proxy for nearest defender when defenders are present.

* Masks: When adding the ball token, extend `player_mask` to length `P+1` and propagate it through training/validation.

* Positional encodings: Use existing `pos_time` and `pos_player` parameters; ensure `pos_player` covers `P+1` when the ball token is present.

## Step 6 — Verification

* Run a short debug training (`Config.DEBUG=True`, `DEBUG_SIZE=1`, `EPOCHS=20`) to ensure shapes are consistent and training proceeds without NaNs.

* Check RMSE logging from `train_all_folds_stt` and ensure `save_fold_artifacts_stt` still writes out models/scalers.

## Expected Impact

* Ball-centric interactions and axial decomposition typically improve stability and reduce noise; anticipate \~0.01–0.02 RMSE improvement over the current 0.562 baseline when trained on full data after tuning.

## Notes

* We do not change loss or regularization in M2; later milestones can add horizon decay tuning or smoothing.

* If needed, we can expand multi-query selection to favor defenders by reading the `is_defense` feature at the last frame; current ordering provides a simple, effective proxy.

