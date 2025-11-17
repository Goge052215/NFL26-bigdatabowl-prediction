import os
import sys
import json

USE_CUDF = False
try:
    # zero/low-code GPU acceleration for DataFrame ops
    os.environ["CUDF_PANDAS_BACKEND"] = "cudf"
    import pandas as pd
    import numpy as np

    USE_CUDF = True
    print("using cuda_backend pandas for faster parallel data processing")
except Exception:
    print("cuda df not used")
    import pandas as pd
    import numpy as np

from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore")

try:
    from config import Config
except ModuleNotFoundError:
    _ensure_module_paths()
    from config import Config
from utilities import (
    set_seed,
    load_input_output,
    write_meta,
    write_cv_log,
    load_saved_ensemble_stt,
)
from preprocess import (
    prepare_sequences_with_advanced_features, 
    prepare_sequences_play_level
)
from model import train_all_folds_stt
from predict import predict_sst

try:
    Config.TIME_TAG = str(TIME_TAG)[:8] + "_" + str(TIME_TAG)[8:]
    Config.SAVE_DIR = Path(f"./output/{Config.TIME_TAG}")
except NameError:
    Config.TIME_TAG = "default"

print(f"Current timetag: {Config.TIME_TAG}")

def train():
    # 1) load training data
    print("\n[1/4] Loading training data")
    train_input, train_output = load_input_output()

    # 2) features + sequences
    print("\n[2/4] Feature Engineering")
    feature_groups = Config.FEATURE_GROUPS
    if getattr(Config, "USE_PLAY_PLAYER_INPUT", False):
        seqs, tdx, tdy, tfids, seq_meta, feat_cols, pmasks = (
            prepare_sequences_play_level(
                train_input,
                output_df=train_output,
                feature_groups=feature_groups,
            )
        )
        input_dim = seqs[0].shape[-1]
        sequences = list(seqs)
        targets_dx = list(tdx)
        targets_dy = list(tdy)
        player_masks = list(pmasks)
    else:
        seqs, tdx, tdy, tfids, seq_meta, feat_cols = (
            prepare_sequences_with_advanced_features(
                train_input,
                output_df=train_output,
                feature_groups=feature_groups,
            )
        )
        input_dim = seqs[0].shape[1]
        sequences = list(seqs)
        targets_dx = list(tdx)
        targets_dy = list(tdy)
        player_masks = None

    # write meta
    write_meta(feat_cols, base_dir=Config.SAVE_DIR)

    # 3) multi-seed × KFold, save per-fold artifacts
    print("\n[3/4] Training model...")
    groups = np.array([f"{d['game_id']}_{d['play_id']}" for d in seq_meta])
    unique_grps, groups = np.unique(groups, return_inverse=True)
    print(f"Created {len(unique_grps)} unique groups (game_play_id) for GroupKFold.")

    seeds = Config.SEEDS
    all_rmse = []
    cv_log = []

    for seed in seeds:
        print(f"\n{'='*70}\n   Seed {seed}\n{'='*70}")
        set_seed(seed)
        gkf = GroupKFold(n_splits=Config.N_FOLDS)

        _all_rmse, _cv_log = train_all_folds_stt(
            gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim, player_masks=player_masks
        )

        all_rmse.extend(_all_rmse)
        cv_log.extend(_cv_log)

    print()
    print(f"[CV SUMMARY] all folds RMSEs: {[f'{r:.4f}' for r in all_rmse]}")
    print(f"[CV SUMMARY] overall mean RMSE = {float(np.mean(all_rmse)):.4f} yards")

    write_cv_log(cv_log, all_rmse)
    return

if __name__ == "__main__":
    if Config.TRAIN:
        train()

# =============================================================================
# Evaluation API Server Setup
# =============================================================================
# New imports for evaluation API
import polars as pl
from utilities import load_saved_ensemble_stt, invert_to_original_direction, build_play_direction_map
from preprocess import prepare_sequences_with_advanced_features
from model import STTransformer
from predict import predict_sst

# Global variables to store models (loaded once on first predict call)
_models_loaded = False
_models = None
_scalers = None
_meta = None
_feature_cols = None

Config.TRAIN = False
Config.SUBMIT = True

def _resolve_models_dir():
    tag = Config.TIME_TAG
    candidates = [
        Path(f"/kaggle/input/nfl2026/{tag}"),
        Path(f"/kaggle/working/output/{tag}"),
        Path(f"./output/{tag}"),
    ]
    for p in candidates:
        if (p / "meta.json").exists():
            return p
    raise FileNotFoundError(
        f"meta.json not found in: {[str(c) for c in candidates]}"
    )

def load_models_once():
    """Load models on first predict call (no 5-minute time limit)"""
    global _models_loaded, _models, _scalers, _meta, _feature_cols

    if _models_loaded:
        return

    print("[SERVER] Loading models for first time...")
    cfg = Config()
    cfg.MODELS_DIR = _resolve_models_dir()

    _models, _scalers, _meta = load_saved_ensemble_stt(cfg.MODELS_DIR, STTransformer)
    _feature_cols = _meta["feature_cols"]

    _models_loaded = True
    print(f"[SERVER] Loaded {len(_models)} models successfully")

    try:
        metrics_path = cfg.MODELS_DIR / "cv_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                m = json.load(f)
            cv_mean = float(m.get("overall_mean_perdim", float("nan")))
            if not np.isnan(cv_mean):
                print(f"[SERVER] Expected CV RMSE (mean): {cv_mean:.4f}")
        else:
            print("[SERVER] CV metrics file not found; LB not available in local testing")
    except Exception:
        pass

def predict(
    test: pl.DataFrame, test_input: pl.DataFrame
) -> pl.DataFrame | pd.DataFrame:
    """
    Inference function: process each batch of data

    Args:
        test: Frames to predict (contains game_id, play_id, nfl_id, frame_id, etc.)
        test_input: Available input data (historical frames)

    Returns:
        DataFrame with x, y coordinates
    """
    global _models, _scalers, _meta, _feature_cols

    # First call: load models (no time limit)
    if not _models_loaded:
        load_models_once()

    # Convert to pandas (our code is pandas-based)
    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()

    cfg = Config()
    saved_groups = _meta.get("feature_groups", cfg.FEATURE_GROUPS)

    # Build sequences
    test_seqs, test_meta, feat_cols_t = prepare_sequences_with_advanced_features(
        test_input_pd,
        test_pd,
        feature_groups=saved_groups,
        required_feature_cols=_feature_cols,
    )

    idx_x = feat_cols_t.index("x")
    idx_y = feat_cols_t.index("y")

    X_test_raw = list(test_seqs)
    x_last_uni = np.array([s[-1, idx_x] for s in X_test_raw], dtype=np.float32)
    y_last_uni = np.array([s[-1, idx_y] for s in X_test_raw], dtype=np.float32)

    all_preds_dx, all_preds_dy = [], []
    for m, sc in zip(_models, _scalers):
        dx_tta, dy_tta = predict_sst(
            m,
            sc,
            X_test_raw,
            cfg.DEVICE,
        )
        all_preds_dx.append(dx_tta)
        all_preds_dy.append(dy_tta)

    ens_dx = np.mean(all_preds_dx, axis=0)
    ens_dy = np.mean(all_preds_dy, axis=0)

    H = ens_dx.shape[1]

    # Build predictions
    rows = []
    tt_idx = test_pd.set_index(["game_id", "play_id", "nfl_id"]).sort_index()

    for i, meta_row in enumerate(test_meta):
        gid = meta_row["game_id"]
        pid = meta_row["play_id"]
        nid = meta_row["nfl_id"]
        play_dir = meta_row["play_direction"]

        try:
            fids = tt_idx.loc[(gid, pid, nid), "frame_id"]
            if isinstance(fids, pd.Series):
                fids = fids.sort_values().tolist()
            else:
                fids = [int(fids)]
        except KeyError:
            continue

        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            x_uni = np.clip(x_last_uni[i] + ens_dx[i, tt], 0, Config.FIELD_X_MAX)
            y_uni = np.clip(y_last_uni[i] + ens_dy[i, tt], 0, Config.FIELD_Y_MAX)
            x_uni, y_uni = invert_to_original_direction(
                x_uni, y_uni, play_dir == "right"
            )
            rows.append({"x": x_uni, "y": y_uni})

    predictions = pl.DataFrame(rows)

    assert len(predictions) == len(test)
    return predictions


if Config.SUBMIT:
    import kaggle_evaluation.nfl_inference_server  # type: ignore

    # Initialize inference server
    inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(
        predict
    )

    # Start server in competition environment
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        print("[SERVER] Starting inference server...")
        inference_server.serve()
    else:
        print("[SERVER] Running local gateway for testing...")
        inference_server.run_local_gateway(
            ("/kaggle/input/nfl-big-data-bowl-2026-prediction/",)
        )