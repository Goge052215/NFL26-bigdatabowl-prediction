import os

USE_CUDF = False
try:
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

from config import Config

import polars as pl
from utilities import load_saved_ensemble_stt, invert_to_original_direction
from preprocess import prepare_sequences_with_advanced_features, prepare_sequences_play_level
from model import STTransformer
from predict import predict_sst

_models_loaded = False
_models = None
_scalers = None
_meta = None
_feature_cols = None

def load_models_once():
    global _models_loaded, _models, _scalers, _meta, _feature_cols

    if _models_loaded:
        return

    print("[SERVER] Loading models for first time...")
    cfg = Config()
    cfg.MODELS_DIR = Path(f"/kaggle/input/nfl2026/{Config.TIME_TAG}")

    _models, _scalers, _meta = load_saved_ensemble_stt(cfg.MODELS_DIR, STTransformer)
    _feature_cols = _meta["feature_cols"]

    _models_loaded = True
    print(f"[SERVER] Loaded {len(_models)} models successfully")


def predict(
    test: pl.DataFrame, test_input: pl.DataFrame
) -> pl.DataFrame | pd.DataFrame:
    global _models, _scalers, _meta, _feature_cols

    if not _models_loaded:
        load_models_once()

    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()

    cfg = Config()
    saved_groups = _meta.get("feature_groups", cfg.FEATURE_GROUPS)

    if getattr(Config, "USE_PLAY_PLAYER_INPUT", False):
        test_seqs, test_meta, feat_cols_t, pmasks = prepare_sequences_play_level(
            test_input_pd,
            test_pd,
            feature_groups=saved_groups,
            required_feature_cols=_feature_cols,
        )
    else:
        test_seqs, test_meta, feat_cols_t = prepare_sequences_with_advanced_features(
            test_input_pd,
            test_pd,
            feature_groups=saved_groups,
            required_feature_cols=_feature_cols,
        )
        pmasks = None

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
            pm_test=pmasks,
        )
        all_preds_dx.append(dx_tta)
        all_preds_dy.append(dy_tta)

    ens_dx = np.mean(all_preds_dx, axis=0)
    ens_dy = np.mean(all_preds_dy, axis=0)

    H = ens_dx.shape[1]

    rows = []
    tt_idx = test_pd.set_index(["game_id", "play_id", "nfl_id"]).sort_index()

    for i, meta_row in enumerate(test_meta):
        gid = meta_row["game_id"]
        pid = meta_row["play_id"]
        nid = meta_row.get("nfl_id") or meta_row.get("target_nfl_id")
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
    inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(
        predict
    )
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        print("[SERVER] Starting inference server...")
        inference_server.serve()
    else:
        print("[SERVER] Running local gateway for testing...")
        inference_server.run_local_gateway(
            ("/kaggle/input/nfl-big-data-bowl-2026-prediction/",)
        )
