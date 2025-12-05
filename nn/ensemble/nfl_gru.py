# %%writefile nfl_gru.py

import os
import shutil


TIMETAG = "20251108_071543"
dest_dir = "./src"
os.makedirs(dest_dir, exist_ok=True)

source_dir = f"/kaggle/input/1114kaiyuan-0562/output/20251108_071543/src"
shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

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


import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore")

from src.config import Config

# =============================================================================
# Evaluation API Server Setup
# =============================================================================
# New imports for evaluation API
import polars as pl
from src.utils import load_saved_ensemble_stt, invert_to_original_direction
from src.preprocess import prepare_sequences_with_advanced_features
from src.model import STTransformer
from src.predict import predict_sst

# Global variables to store models (loaded once on first predict call)
_models_loaded = False
_models = None
_scalers = None
_meta = None
_feature_cols = None


def load_models_once():
    """Load models on first predict call (no 5-minute time limit)"""
    global _models_loaded, _models, _scalers, _meta, _feature_cols

    if _models_loaded:
        return

    print("[SERVER] Loading models for first time...")
    cfg = Config()
    cfg.MODELS_DIR = Path(f"/kaggle/input/1114kaiyuan-0562/output/20251108_071543")

    _models, _scalers, _meta = load_saved_ensemble_stt(cfg.MODELS_DIR, STTransformer)
    _feature_cols = _meta["feature_cols"]

    _models_loaded = True
    print(f"[SERVER] Loaded {len(_models)} models successfully")


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
