import json, joblib
import os, random, torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn

from config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_input_output():
    base_dir = Path(str(Config.DATA_DIR))
    train_input_files = [
        base_dir / f"train/input_2023_w{w:02d}.csv"
        for w in range(1, 19 if not Config.DEBUG else 1 + Config.DEBUG_SIZE)
    ]
    train_output_files = [
        base_dir / f"train/output_2023_w{w:02d}.csv"
        for w in range(1, 19 if not Config.DEBUG else 1 + Config.DEBUG_SIZE)
    ]
    train_input = pd.concat(
        [pd.read_csv(f) for f in train_input_files if f.exists()], ignore_index=True
    )
    train_output = pd.concat(
        [pd.read_csv(f) for f in train_output_files if f.exists()], ignore_index=True
    )

    # Filter out outliers to better align CV with LB
    # Ref: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/611647#3310487
    bad_game_id = 2023091100
    bad_play_id = 3167

    before_in = len(train_input)
    before_out = len(train_output)

    train_input = train_input[
        ~(
            (train_input["game_id"] == bad_game_id)
            & (train_input["play_id"] == bad_play_id)
        )
    ]
    train_output = train_output[
        ~(
            (train_output["game_id"] == bad_game_id)
            & (train_output["play_id"] == bad_play_id)
        )
    ]

    print("Filtered input rows: ", before_in - len(train_input))
    print("Filtered output rows: ", before_out - len(train_output))

    return train_input, train_output


def wrap_angle_deg(s):
    # Map to (-180, 180]
    return ((s + 180.0) % 360.0) - 180.0


def build_play_direction_map(df_in: pd.DataFrame) -> pd.Series:
    return (
        df_in[["game_id", "play_id", "play_direction"]]
        .drop_duplicates()
        .set_index(["game_id", "play_id"])["play_direction"]
    )


def unify_left_direction_ipt(df: pd.DataFrame) -> pd.DataFrame:
    if "play_direction" not in df.columns:
        return df

    df = df.copy()
    right = df["play_direction"].eq("right")

    if "x" in df.columns:
        df.loc[right, "x"] = Config.FIELD_X_MAX - df.loc[right, "x"]
    if "y" in df.columns:
        df.loc[right, "y"] = Config.FIELD_Y_MAX - df.loc[right, "y"]

    if "ball_land_x" in df.columns:
        df.loc[right, "ball_land_x"] = Config.FIELD_X_MAX - df.loc[right, "ball_land_x"]
    if "ball_land_y" in df.columns:
        df.loc[right, "ball_land_y"] = Config.FIELD_Y_MAX - df.loc[right, "ball_land_y"]

    for col in ("dir", "o"):
        if col in df.columns:
            df.loc[right, col] = (df.loc[right, col].astype(float) + 180.0) % 360.0

    return df


def unify_left_direction_opt(df: pd.DataFrame, dir_map: dict) -> pd.DataFrame:
    df["play_direction"] = df.apply(
        lambda r: dir_map.get((r["game_id"], r["play_id"])), axis=1
    )
    right = df["play_direction"].eq("right")

    if "x" in df.columns:
        df.loc[right, "x"] = Config.FIELD_X_MAX - df.loc[right, "x"]
    if "y" in df.columns:
        df.loc[right, "y"] = Config.FIELD_Y_MAX - df.loc[right, "y"]

    df.drop(columns=["play_direction"], inplace=True)

    return df


def invert_to_original_direction(x_u, y_u, play_dir_right: bool):
    """Invert unified (left) coordinates back to original play direction."""
    if not play_dir_right:
        return float(x_u), float(y_u)
    return float(Config.FIELD_X_MAX - x_u), float(Config.FIELD_Y_MAX - y_u)


def _seed_dir(base_dir: Path, seed: int) -> Path:
    d = base_dir / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class MinimalScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / (self.scale_ + 1e-8)


def save_fold_artifacts_stt(
    seed: int, fold: int, scaler, model: nn.Module, base_dir: Path
):
    sdir = _seed_dir(base_dir, seed)
    np.savez(sdir / f"scaler_fold{fold}.npz", mean=scaler.mean_, scale=scaler.scale_)
    torch.save(model.state_dict(), sdir / f"model_fold{fold}.pt")


def write_meta(feature_cols: list, base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seeds": Config.SEEDS,
        "n_folds": Config.N_FOLDS,
        "feature_cols": feature_cols,
        "window_size": Config.WINDOW_SIZE,
        "feature_groups": Config.FEATURE_GROUPS,
        "version": 1,
    }
    with open(base_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] wrote meta.json to {base_dir}")


def write_cv_log(cv_log: list, all_rmse: list):
    with open(Config.SAVE_DIR / "cv_metrics.json", "w") as f:
        json.dump(
            {
                "per_fold": cv_log,
                "overall_mean_perdim": float(np.mean(all_rmse)),
            },
            f,
            indent=2,
        )
    print(f"\nCV metrics written to {Config.SAVE_DIR / 'cv_metrics.json'}")


def load_saved_ensemble_stt(base_dir: Path, model_class: torch.nn.Module):
    meta_path = base_dir / "meta.json"
    assert meta_path.exists(), f"meta.json not found: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    seeds = meta["seeds"]
    n_folds = int(meta["n_folds"])

    models, scalers = [], []
    for seed in seeds:
        sdir = base_dir / f"seed_{seed}"
        for fold in range(1, n_folds + 1):
            sc_npz = sdir / f"scaler_fold{fold}.npz"
            sc_path = sdir / f"scaler_fold{fold}.pkl"
            model_path = sdir / f"model_fold{fold}.pt"
            if not (sc_path.exists() and model_path.exists()):
                if not (sc_npz.exists() and model_path.exists()):
                    print(f"[WARN] missing seed={seed} fold={fold}, skip")
                    continue
            if sc_npz.exists():
                data = np.load(sc_npz)
                scaler = MinimalScaler(data["mean"], data["scale"]) 
            else:
                try:
                    obj = joblib.load(sc_path)
                    mean = getattr(obj, "mean_", None)
                    scale = getattr(obj, "scale_", None)
                    if mean is not None and scale is not None:
                        scaler = MinimalScaler(mean, scale)
                    else:
                        scaler = obj
                except Exception:
                    print(f"[WARN] failed to load scaler for seed={seed} fold={fold}")
                    continue
            m = model_class(len(feature_cols)).to(Config.DEVICE)
            m.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            m.eval()
            scalers.append(scaler)
            models.append(m)

    return models, scalers, meta


def prepare_targets_stt(batch_dx, batch_dy, max_h):
    tensors_x, tensors_y, masks = [], [], []

    for dx, dy in zip(batch_dx, batch_dy):
        L = len(dx)
        padded_x = np.pad(dx, (0, max_h - L), constant_values=0).astype(np.float32)
        padded_y = np.pad(dy, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0

        tensors_x.append(torch.tensor(padded_x))
        tensors_y.append(torch.tensor(padded_y))
        masks.append(torch.tensor(mask))

    targets = torch.stack([torch.stack(tensors_x), torch.stack(tensors_y)], dim=-1)
    return targets, torch.stack(masks)