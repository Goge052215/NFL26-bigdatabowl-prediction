#%%writefile nfl_gru2.py

import os
import shutil


TIMETAG = "20251108_071543"
dest_dir = "./src"
os.makedirs(dest_dir, exist_ok=True)

source_dir = f"/kaggle/input/1113gru-0576/output/20251108_071543/src"
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
import torch
import joblib

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
_mode = None


def load_models_once():
    """Load models on first predict call (no 5-minute time limit)"""
    global _models_loaded, _models, _scalers, _meta, _feature_cols, _mode

    if _models_loaded:
        return

    print("[SERVER] Loading models for first time...")
    cfg = Config()
    cfg.MODELS_DIR = Path(f"/kaggle/input/1113gru-0576/output/20251108_071543")

    try:
        _models, _scalers, _meta = load_saved_ensemble_stt(cfg.MODELS_DIR, STTransformer)
        _feature_cols = _meta["feature_cols"]
        _mode = "stt"
        _models_loaded = True
        print(f"[SERVER] Loaded {len(_models)} STT models successfully")
        return
    except Exception as e:
        print(f"[SERVER] STT load failed: {e}")

    models_x, models_y, scalers, meta = load_saved_ensemble_gru(cfg.MODELS_DIR)
    _models = (models_x, models_y)
    _scalers = scalers
    _meta = meta
    _feature_cols = _meta["feature_cols"]
    _mode = "gru"
    _models_loaded = True
    print(f"[SERVER] Loaded {len(models_x)} GRU models successfully")


def load_saved_ensemble_gru(base_dir: Path):
    meta_path = base_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_cols", [])
    seeds = meta.get("seeds", [])
    n_folds = int(meta.get("n_folds", 0))
    horizon = int(meta.get("max_future_horizon", Config.MAX_FUTURE_HORIZON))
    hidden_dim = int(meta.get("hidden_dim", 128))
    bidirectional = bool(meta.get("bidirectional", False))

    try:
        from src.model import SeqModel as GRUModel
    except Exception:
        GRUModel = None

    assert GRUModel is not None, "SeqModel not found in src.model"

    models_x, models_y, scalers = [], [], []
    for seed in seeds:
        sdir = base_dir / f"seed_{seed}"
        for fold in range(1, n_folds + 1):
            sc_path = sdir / f"scaler_fold{fold}.pkl"
            dx_path = sdir / f"model_dx_fold{fold}.pt"
            dy_path = sdir / f"model_dy_fold{fold}.pt"
            if not (sc_path.exists() and dx_path.exists() and dy_path.exists()):
                continue
            scaler = joblib.load(sc_path)
            mx = GRUModel(len(feature_cols), horizon, hidden_dim=hidden_dim, bidirectional=bidirectional).to(Config.DEVICE)
            mx.load_state_dict(torch.load(dx_path, map_location=Config.DEVICE))
            mx.eval()
            my = GRUModel(len(feature_cols), horizon, hidden_dim=hidden_dim, bidirectional=bidirectional).to(Config.DEVICE)
            my.load_state_dict(torch.load(dy_path, map_location=Config.DEVICE))
            my.eval()
            scalers.append(scaler)
            models_x.append(mx)
            models_y.append(my)

    assert len(models_x) > 0, f"No GRU models loaded from {base_dir}"
    return models_x, models_y, scalers, meta


def predict_gru_single(mx, my, scaler, X_test_raw, device):
    base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)
    with torch.no_grad():
        dx = mx(xt)
        dy = my(xt)
    return dx.detach().cpu().numpy(), dy.detach().cpu().numpy()


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
    if _mode == "stt":
        for m, sc in zip(_models, _scalers):
            dx_tta, dy_tta = predict_sst(
                m,
                sc,
                X_test_raw,
                cfg.DEVICE,
            )
            all_preds_dx.append(dx_tta)
            all_preds_dy.append(dy_tta)
    else:
        models_x, models_y = _models
        for mx, my, sc in zip(models_x, models_y, _scalers):
            dx_tta, dy_tta = predict_gru_single(mx, my, sc, X_test_raw, cfg.DEVICE)
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
