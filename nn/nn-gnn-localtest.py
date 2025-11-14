# Leaderboard score: 0.562 RMSE

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
import random
import time
import json
import joblib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# === Configuration ===

class Config:
    # Local dataset
    DATA_DIR = Path("/Users/goge/nfl26/data")
    TIME_TAG = "default"

    # Status flag
    # Force LOCAL mode (no Kaggle integration)
    TRAIN = True
    SUBMIT = False

    # Debug mode: check pipeline integrity
    DEBUG = False
    DEBUG_SIZE = 1

    OUTPUT_DIR = Path("./output")
    SAVE_DIR = Path(f"./output/{TIME_TAG}")

    # Local workers
    MAX_WORKER = min(8, os.cpu_count() or 1)

    # Specify the feature group
    FEATURE_GROUPS = [
        "target_alignment",
        "lag",
        "distance_rate",
        "time",
        "role",
        "passer",
        "curvature",
        "route",
        "receiver",
        "neighbor_gnn",
    ]

    # Neighbors feature
    K_NEIGH = 3
    RADIUS = 28.0
    TAU = 8.0

    # Training Setting
    SEEDS = [42]
    # SEEDS = [42, 19, 89, 64]
    N_FOLDS = 5
    BATCH_SIZE = 256
    EPOCHS = 200 if not DEBUG else 20
    PATIENCE = 30
    LEARNING_RATE = 1e-3

    WINDOW_SIZE = 10
    MAX_PLAYER = 9
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 55  # Number of steps to predict (filter out 94)

    N_HEADS = 4
    N_LAYERS = 2
    MLP_HIDDEN_DIM = 256
    N_RES_BLOCKS = 2
    N_QUERYS = 3
    ADD_BALL_TOKEN = True
    USE_AXIAL_ATTENTION = True

    # Device selection: prefer Apple MPS, otherwise CPU (force local training)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # Field Setting
    YARDS_TO_METERS = 0.9144
    FPS = 10.0

    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

    # Input/Model options
    # When True, build play-level sequences with explicit player axis: [N_play, N_player, T_in, dim]
    USE_PLAY_PLAYER_INPUT = True

# === Utilities ===

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Guard CUDA seed to avoid legacy GPU dependency
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_input_output():
    train_input_files = [
        Config.DATA_DIR / f"train/input_2023_w{w:02d}.csv"
        for w in range(1, 19 if not Config.DEBUG else 1 + Config.DEBUG_SIZE)
    ]
    train_output_files = [
        Config.DATA_DIR / f"train/output_2023_w{w:02d}.csv"
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


def save_fold_artifacts_stt(
    seed: int, fold: int, scaler, model: nn.Module, base_dir: Path
):
    sdir = _seed_dir(base_dir, seed)
    joblib.dump(scaler, sdir / f"scaler_fold{fold}.pkl")
    torch.save(model.state_dict(), sdir / f"model_fold{fold}.pt")


def write_meta(feature_cols: list, base_dir: Path):
    # Ensure the base directory exists before writing metadata
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
    # Ensure the save directory exists before writing CV metrics
    Config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
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
            sc_path = sdir / f"scaler_fold{fold}.pkl"
            model_path = sdir / f"model_fold{fold}.pt"
            if not (sc_path.exists() and model_path.exists()):
                print(f"[WARN] missing seed={seed} fold={fold}, skip")
                continue
            scaler = joblib.load(sc_path)
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


# === Feature Engineering ===
class FeatureEngineer:
    """
    Modular, ablation-friendly feature builder.
    """

    def __init__(self, feature_groups_to_create: list):
        self.gcols = ["game_id", "play_id", "nfl_id"]
        self.active_groups = feature_groups_to_create
        # Map feature groups to (function, interactive flag)
        # interactive = True: feature requires information from other players
        # interactive = False: can be computed using only the local subset of input_df
        self.feature_creators = {
            "target_alignment": (self._create_target_alignment_features, False),
            "multi_window": (self._create_multi_window_features, False),
            "lag": (self._create_extended_lag_features, False),
            "motion_change": (self._create_motion_change_features, False),
            "field_position": (self._create_field_position_features, False),
            "distance_rate": (self._create_distance_rate_features, False),
            "geometric": (self._create_geometric_features, False),
            "neighbor_gnn": (self._create_neighbor_features, True),
            "time": (self._create_time_features, False),
            "role": (self._create_role_features, False),
            "passer": (self._create_passer_features, True),
            "curvature": (self._create_curvature_features, False),
            "route": (self._create_route_features, False),
            "receiver": (self._create_receiver_features, True),
        }
        self.created_feature_cols = []

    def _height_to_feet(self, height_str):
        try:
            ft, inches = map(int, str(height_str).split("-"))
            return ft + inches / 12
        except Exception:
            return 6.0

    def _mirror_angle(self, df: pd.DataFrame, cols: list):
        for col in cols:
            df[col] = (450 - df[col]) % 360
        return df

    def _warp_angle(self, df: pd.DataFrame, col: str):
        return np.minimum(df[col], 360 - df[col])

    def _create_basic_features(self, df: pd.DataFrame):
        """Simple derived features from original columns"""
        # Convert angle from dataset convention to standard Cartesian coordinates
        angle_cols = ["dir", "o"]
        df = self._mirror_angle(df, angle_cols)

        # Height & Weight & BMI
        df["player_height_feet"] = df["player_height"].apply(self._height_to_feet)
        height_parts = df["player_height"].str.split("-", expand=True)
        df["height_inches"] = height_parts[0].astype(float) * 12 + height_parts[
            1
        ].astype(float)
        df["bmi"] = (df["player_weight"] / (df["height_inches"] ** 2)) * 703

        # Velocity & Acceleration & Momentum
        dir_rad = np.deg2rad(df["dir"].fillna(0))
        df["velocity_x"] = df["s"] * np.cos(dir_rad)
        df["velocity_y"] = df["s"] * np.sin(dir_rad)
        # NOTE: acceleration_x/y may be incorrect
        df["acceleration_x"] = df["a"] * np.cos(dir_rad)
        df["acceleration_y"] = df["a"] * np.sin(dir_rad)

        df["momentum_x"] = df["velocity_x"] * df["player_weight"]
        df["momentum_y"] = df["velocity_y"] * df["player_weight"]
        df["speed_squared"] = df["s"] ** 2
        df["kinetic_energy"] = 0.5 * df["player_weight"] * df["speed_squared"]

        # TODO: Consider direction
        df["orientation_diff"] = np.abs(df["o"] - df["dir"])
        df["orientation_diff"] = self._warp_angle(df, "orientation_diff")

        # Play direction (1 = left, 0 = right)
        df["play_direction"] = (df["play_direction"] == "left").astype(int)

        # Player side
        df["is_offense"] = (df["player_side"] == "Offense").astype(int)
        df["is_defense"] = (df["player_side"] == "Defense").astype(int)
        # Player role
        df["is_receiver"] = (df["player_role"] == "Targeted Receiver").astype(int)
        df["is_coverage"] = (df["player_role"] == "Defensive Coverage").astype(int)
        df["is_passer"] = (df["player_role"] == "Passer").astype(int)

        # Ball
        ball_dx = df["ball_land_x"] - df["x"]
        ball_dy = df["ball_land_y"] - df["y"]
        df["distance_to_ball"] = np.sqrt(ball_dx**2 + ball_dy**2)
        df["angle_to_ball"] = np.arctan2(ball_dy, ball_dx)
        df["ball_direction_x"] = ball_dx / (df["distance_to_ball"] + 1e-6)
        df["ball_direction_y"] = ball_dy / (df["distance_to_ball"] + 1e-6)
        df["angle_diff"] = np.abs(df["o"] - np.degrees(df["angle_to_ball"]))
        df["angle_diff"] = self._warp_angle(df, "angle_diff")
        df["closing_speed"] = (
            df["velocity_x"] * df["ball_direction_x"]
            + df["velocity_y"] * df["ball_direction_y"]
        )

        base = [
            # Original
            "x",
            "y",
            "s",
            "a",
            "o",
            "dir",
            "frame_id",
            "ball_land_x",
            "ball_land_y",
            "player_weight",
            # Derived
            "player_height_feet",
            "bmi",
            "velocity_x",
            "velocity_y",
            "acceleration_x",
            "acceleration_y",
            "momentum_x",
            "momentum_y",
            "speed_squared",
            "kinetic_energy",
            "orientation_diff",
            # "play_direction",
            "is_offense",
            "is_defense",
            "is_receiver",
            "is_coverage",
            "is_passer",
            "distance_to_ball",
            "angle_to_ball",
            "ball_direction_x",
            "ball_direction_y",
            "angle_diff",
            "closing_speed",
        ]
        self.created_feature_cols.extend([c for c in base if c in df.columns])
        return df

    def _create_target_alignment_features(self, df: pd.DataFrame):
        """
        Compute alignment features between a player's movement vector and the ball's direction.

        These features describe how the player's motion aligns with the target (ball) and can help
        predict actions such as approaching, intercepting, or moving away from the ball.
        """
        new_cols = []
        if not {"ball_direction_x", "ball_direction_y"}.issubset(df.columns):
            return df, new_cols

        # Velocity
        if {"velocity_x", "velocity_y"}.issubset(df.columns):
            df["velocity_alignment"] = (
                df["velocity_x"] * df["ball_direction_x"]
                + df["velocity_y"] * df["ball_direction_y"]
            )
            df["velocity_perpendicular"] = (
                df["velocity_x"] * (-df["ball_direction_y"])
                + df["velocity_y"] * df["ball_direction_x"]
            )
            new_cols.extend(["velocity_alignment", "velocity_perpendicular"])

        # Acceleration
        if {"acceleration_x", "acceleration_y"}.issubset(df.columns):
            df["accel_alignment"] = (
                df["acceleration_x"] * df["ball_direction_x"]
                + df["acceleration_y"] * df["ball_direction_y"]
            )
            df["accel_perpendicular"] = (
                df["acceleration_x"] * (-df["ball_direction_y"])
                + df["acceleration_y"] * df["ball_direction_x"]
            )
            new_cols.extend(["accel_alignment", "accel_perpendicular"])

        return df, new_cols

    def _create_multi_window_features(self, df: pd.DataFrame):
        new_cols = []
        mask = df["player_to_predict"]

        df_target = df.loc[mask].copy()

        for window in (3, 5, 10, 20):
            for col in ("velocity_x", "velocity_y", "s"):
                if col in df.columns:
                    r_mean = (
                        df_target.groupby(self.gcols)[col]
                        .rolling(window, min_periods=1)
                        .mean()
                        .reset_index(level=list(range(len(self.gcols))), drop=True)
                    )
                    r_std = (
                        df_target.groupby(self.gcols)[col]
                        .rolling(window, min_periods=1)
                        .std()
                        .reset_index(level=list(range(len(self.gcols))), drop=True)
                    )

                    df.loc[mask, f"{col}_roll{window}"] = r_mean
                    df.loc[mask, f"{col}_std{window}"] = r_std.fillna(0.0)
                    df.loc[mask, f"{col}_dev{window}"] = (
                        df.loc[mask, col].values - r_mean.values
                    )

                    new_cols.extend(
                        [
                            f"{col}_roll{window}",
                            f"{col}_std{window}",
                            f"{col}_dev{window}",
                        ]
                    )

        # speed_trend_ratio
        if "s_roll3" in df.columns and "s_roll20" in df.columns:
            df.loc[mask, "speed_trend_ratio"] = df.loc[mask, "s_roll3"] / (
                df.loc[mask, "s_roll20"] + 1e-3
            )
            new_cols.append("speed_trend_ratio")

        return df, new_cols

    def _create_extended_lag_features(self, df: pd.DataFrame):
        new_cols = []
        mask = df["player_to_predict"]
        df_target = df.loc[mask].copy()

        for lag in (1, 2, 3):
            for col in ("velocity_x", "velocity_y", "s"):
                if col in df.columns:
                    g = df_target.groupby(self.gcols)[col]

                    lagv = g.shift(lag)
                    fillv = lagv.fillna(g.transform("first"))

                    df.loc[mask, f"{col}_lag{lag}"] = fillv[mask]
                    new_cols.append(f"{col}_lag{lag}")

                    diffv = df[col] - fillv
                    df.loc[mask, f"{col}_diff_lag{lag}"] = diffv[mask]
                    new_cols.append(f"{col}_diff_lag{lag}")

        return df, new_cols

    def _create_motion_change_features(self, df: pd.DataFrame):
        """
        Compute features representing changes in a player's velocity, speed, and movement direction between consecutive time steps.
        """
        new_cols = []
        diff_cols = [
            "velocity_x",
            "velocity_y",
            "s",
            "a",
            "dir",
            "o",
            # "angle_to_ball",
        ]

        for col in diff_cols:
            if col not in df.columns:
                print(f"[WARNING]: {col} not in columns of df!")
                continue

            new_col = f"{col}_change"
            df[new_col] = df.groupby(self.gcols)[col].diff().fillna(0.0)
            if col in ["dir", "o"]:
                df[new_col] = wrap_angle_deg(df[new_col])

            new_cols.append(new_col)

        return df, new_cols

    def _create_field_position_features(self, df: pd.DataFrame):
        df["dist_from_left"] = df["y"]
        df["dist_from_right"] = Config.FIELD_X_MAX - df["y"]
        df["dist_from_sideline"] = np.minimum(
            df["dist_from_left"], df["dist_from_right"]
        )
        df["dist_from_endzone"] = np.minimum(df["x"], Config.FIELD_Y_MAX - df["x"])
        df["field_zone_x"] = (df["x"] / Config.FIELD_Y_MAX * 5).astype(int).clip(0, 4)
        df["field_zone_y"] = (df["y"] / Config.FIELD_X_MAX * 3).astype(int).clip(0, 2)
        df["in_red_zone"] = (df["dist_from_endzone"] < 20).astype(np.int8)
        df["near_sideline"] = (df["dist_from_sideline"] < 5).astype(np.int8)
        df["dist_from_center"] = np.hypot(
            df["x"] - Config.FIELD_Y_MAX / 2, df["y"] - Config.FIELD_X_MAX / 2
        )
        return df, [
            "dist_from_sideline",
            "dist_from_endzone",
            "field_zone_x",
            "field_zone_y",
            "in_red_zone",
            "near_sideline",
            "dist_from_center",
        ]

    def _create_distance_rate_features(self, df: pd.DataFrame):
        """Features related to distance to ball"""
        new_cols = []
        if "distance_to_ball" in df.columns:
            d = df.groupby(self.gcols)["distance_to_ball"].diff()
            df["d2ball_dt"] = d.fillna(0.0) * Config.FPS
            df["d2ball_ddt"] = (
                df.groupby(self.gcols)["d2ball_dt"].diff().fillna(0.0) * Config.FPS
            )
            df["time_to_intercept"] = (
                df["distance_to_ball"] / (df["d2ball_dt"].abs() + 1e-3)
            ).clip(0, 10)
            new_cols.extend(["d2ball_dt", "d2ball_ddt", "time_to_intercept"])
        return df, new_cols

    def _create_geometric_features(self, df: pd.DataFrame):
        new_cols = []
        t_total = df["num_frames_output"] / Config.FPS

        # Estimate endpoint based on current status
        df["geo_endpoint_x"] = df["x"] + df["velocity_x"] * t_total
        df["geo_endpoint_y"] = df["y"] + df["velocity_y"] * t_total
        df["geo_endpoint_x"] = df["geo_endpoint_x"].clip(
            Config.FIELD_X_MIN, Config.FIELD_X_MAX
        )
        df["geo_endpoint_y"] = df["geo_endpoint_y"].clip(
            Config.FIELD_Y_MIN, Config.FIELD_Y_MAX
        )
        new_cols.extend(["geo_endpoint_x", "geo_endpoint_y"])

        # TODO: Mirror Receiver

        return df, new_cols

    # NOTE: The neighbor feature is IMPORTANT for model without Spatio info
    def _create_neighbor_features(self, df: pd.DataFrame):
        new_cols = []
        info_cols = [
            "frame_id",
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "player_side",
            # "bmi",
            # "momentum_x",
            # "momentum_y",
            # "kinetic_energy",
            "dir",
            # "o",
            "player_to_predict",
        ]

        # Extract features for last frame
        info_df = df[self.gcols + info_cols].copy()

        last_df = (
            info_df[info_df["player_to_predict"]]
            .sort_values(self.gcols + ["frame_id"])
            .groupby(self.gcols, as_index=False)
            .tail(1)
            .rename(columns={"frame_id": "frame_id_last"})
            .reset_index(drop=True)
        )

        nb_cols_map = {c: f"{c}_nb" for c in info_cols + ["nfl_id"]}
        info_df = last_df.merge(
            info_df.rename(columns=nb_cols_map),
            left_on=["game_id", "play_id", "frame_id_last"],
            right_on=["game_id", "play_id", "frame_id_nb"],
            how="left",
        )

        info_df.drop(
            columns=["player_to_predict", "player_to_predict_nb"], inplace=True
        )
        info_df = info_df[info_df["nfl_id_nb"] != info_df["nfl_id"]]

        # Calculate distance and diff of velocity between player and neighbors
        dx = info_df["x_nb"] - info_df["x"]
        dy = info_df["y_nb"] - info_df["y"]
        info_df["dx"] = dx
        info_df["dy"] = dy

        info_df["dvx"] = info_df["velocity_x_nb"] - info_df["velocity_x"]
        info_df["dvy"] = info_df["velocity_y_nb"] - info_df["velocity_y"]

        info_df["dist"] = np.sqrt(info_df["dx"] ** 2 + info_df["dy"] ** 2)
        info_df = info_df[np.isfinite(info_df["dist"]) & (info_df["dist"] > 1e-6)]
        info_df = info_df[info_df["dist"] <= Config.RADIUS]

        # Calculate weight based on distance
        info_df["rnk"] = (
            info_df.groupby(self.gcols)["dist"].rank(method="first").astype(int)
        )
        info_df = info_df[info_df["rnk"] <= Config.K_NEIGH]

        info_df["w"] = np.exp(-info_df["dist"] / float(Config.TAU))
        sum_w = info_df.groupby(self.gcols)["w"].transform("sum")
        info_df["wn"] = np.where(sum_w > 0, info_df["w"] / sum_w, 0.0)

        info_df["is_ally"] = (
            info_df["player_side_nb"] == info_df["player_side"]
        ).astype(np.float32)
        info_df["is_opp"] = 1.0 - info_df["is_ally"]

        info_df["wn_ally"] = info_df["wn"] * info_df["is_ally"]
        info_df["wn_opp"] = info_df["wn"] * (1.0 - info_df["is_ally"])

        # Create weight col for agg cols
        orig_agg_cols = [
            "momentum_x",
            "momentum_y",
            "kinetic_energy",
            "bmi",
        ]

        diff_agg_cols = [
            "dx",
            "dy",
            "dvx",
            "dvy",
        ]

        for col in orig_agg_cols + diff_agg_cols:
            if col in orig_agg_cols and col not in info_cols:
                continue
            col_nb = f"{col}_nb" if col in orig_agg_cols else col
            info_df[f"{col}_ally_w"] = info_df[col_nb] * info_df["wn_ally"]
            info_df[f"{col}_opp_w"] = info_df[col_nb] * info_df["wn_opp"]

        # ally / opp distance
        info_df["dist_ally"] = np.where(
            info_df["is_ally"] > 0.5, info_df["dist"], np.nan
        )
        info_df["dist_opp"] = np.where(
            info_df["is_ally"] < 0.5, info_df["dist"], np.nan
        )

        # Aggregation
        agg_dict = {}
        for col in orig_agg_cols + diff_agg_cols:
            if col in orig_agg_cols and col not in info_cols:
                continue
            agg_dict[f"{col}_ally_w"] = "sum"
            agg_dict[f"{col}_opp_w"] = "sum"
        agg_dict.update(
            {
                "is_ally": "sum",
                "is_opp": "sum",
                "dist_ally": ["min", "mean"],
                "dist_opp": ["min", "mean"],
            }
        )

        ag = info_df.groupby(self.gcols).agg(agg_dict)
        # Flatten MultiIndex columns
        ag.columns = ["_".join(filter(None, col)).strip() for col in ag.columns.values]
        ag = ag.reset_index()

        # Nearest neighbors
        ADD_NEAREST_FEAT = True
        if ADD_NEAREST_FEAT:
            K = 3
            near_cols = ["dist"]
            # near_cols = ["dist", "x", "y", "dir"]
            near = info_df.loc[
                info_df["rnk"] <= K, self.gcols + ["rnk"] + near_cols
            ].copy()
            # near_cols = ["dist", "x", "y", "dir"]
            # near = info_df_all.loc[
            #     info_df_all["rnk"] <= K, self.gcols + ["rnk"] + near_cols
            # ].copy()
            # near["rnk"] = near["rnk"].astype(int)

            for col in near_cols:
                dwide = near.pivot_table(
                    index=self.gcols, columns="rnk", values=col, aggfunc="first"
                )
                dwide = dwide.rename(
                    columns={i: f"gnn_n{int(i)}_{col}" for i in dwide.columns}
                ).reset_index()
                ag = ag.merge(dwide, on=self.gcols, how="left")

        # Merge back to df
        new_cols = [c for c in ag.columns if c not in self.gcols]
        for c in new_cols:
            ag[c] = ag[c].fillna(0.0)

        df = df.merge(ag, on=self.gcols, how="left")

        # Defense(Offense) Pressure
        ADD_PRESSURE_FEAT = True
        if ADD_PRESSURE_FEAT:
            df["dist_opp_min"] = df["dist_opp_min"].replace(0, np.nan)
            df["dist_ally_min"] = df["dist_ally_min"].replace(0, np.nan)
            df["dist_opp_eff"] = df["dist_opp_min"].fillna(np.inf)
            df["dist_ally_eff"] = df["dist_ally_min"].fillna(np.inf)

            df["pressure"] = 1 / np.maximum(df["dist_opp_eff"], 0.5)
            df["under_pressure"] = (df["dist_opp_eff"] < 3).astype(int)
            df["have_assistance"] = (
                (df["dist_ally_min"].notna())
                & (df["dist_ally_eff"] < df["dist_opp_eff"])
            ).astype(int)

            df.drop(columns=["dist_opp_eff", "dist_ally_eff"], inplace=True)

            df["pressure_speed"] = df["pressure"] * df["s"]
            df["ally_density"] = df["is_ally_sum"] / (
                np.pi * df["dist_ally_mean"] ** 2 + 1e-6
            )
            df["oppn_density"] = df["is_opp_sum"] / (
                np.pi * df["dist_opp_mean"] ** 2 + 1e-6
            )
            df["density_ratio"] = df["ally_density"] / (df["oppn_density"] + 1e-6)

            new_cols.extend(
                [
                    "pressure",
                    "under_pressure",
                    "have_assistance",
                    "pressure_speed",
                    "ally_density",
                    "oppn_density",
                    "density_ratio",
                ]
            )

        return df, new_cols

    def _create_time_features(self, df: pd.DataFrame):
        new_cols = []

        max_frame = df.groupby(self.gcols)["frame_id"].transform("max")
        df["time_to_end"] = max_frame - df["frame_id"] + df["num_frames_output"]
        df["time_urgency"] = 1 / df["time_to_end"]
        df["time_dist_urgency"] = df["distance_to_ball"] / df["time_to_end"]
        new_cols.extend(["time_to_end", "time_urgency", "time_dist_urgency"])

        df["time_normalized_pass"] = df["frame_id"] / max_frame
        df["time_normalized_all"] = df["frame_id"] / (
            max_frame + df["num_frames_output"]
        )
        new_cols.extend(["time_normalized_pass", "time_normalized_all"])

        return df, new_cols

    def _create_opponent_features(self, df: pd.DataFrame):
        new_cols = []
        return df, new_cols

    def _create_role_features(self, df: pd.DataFrame):
        new_cols = []
        if {"is_receiver", "velocity_alignment"}.issubset(df.columns):
            df["receiver_optimality"] = df["is_receiver"] * df["velocity_alignment"]
            df["receiver_deviation"] = df["is_receiver"] * np.abs(
                df.get("velocity_perpendicular", 0.0)
            )
            df["receiver_speed_usage"] = (
                df["is_receiver"] * df["s"] / (df["s"].max() + 1e-3)
            )
            new_cols.extend(
                ["receiver_optimality", "receiver_deviation", "receiver_speed_usage"]
            )
        if {"is_coverage", "closing_speed"}.issubset(df.columns):
            df["defender_closing_speed"] = df["is_coverage"] * df["closing_speed"]
            df["defender_pressure"] = df["is_coverage"] / (
                df.get("distance_to_ball", 10.0) + 1e-3
            )
            new_cols.extend(["defender_closing_speed", "defender_pressure"])

        return df, new_cols

    def _create_passer_features(self, df: pd.DataFrame):
        # Get (x, y) position of passer
        passer_df = (
            df[df["player_role"] == "Passer"]
            .groupby(["game_id", "play_id", "frame_id"], as_index=False)[["x", "y"]]
            .first()
            .rename(columns={"x": "passer_x", "y": "passer_y"})
        )

        # Merge
        df = df.merge(
            passer_df,
            on=["game_id", "play_id", "frame_id"],
            how="left",
            validate="many_to_one",
        )

        mask = df["player_to_predict"]

        dx = df.loc[mask, "x"].astype("float32") - df.loc[mask, "passer_x"].astype(
            "float32"
        )
        dy = df.loc[mask, "y"].astype("float32") - df.loc[mask, "passer_y"].astype(
            "float32"
        )

        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        ux, uy = dx / dist, dy / dist

        vx = df.loc[mask, "velocity_x"].astype("float32")
        vy = df.loc[mask, "velocity_y"].astype("float32")

        align = vx * ux + vy * uy
        perp = vx * (-uy) + vy * ux

        dir_rad = np.deg2rad(df.loc[mask, "dir"].fillna(0).astype("float32"))

        # bearing
        to_passer_angle = np.arctan2(-dy, -dx)
        bearing = np.rad2deg(to_passer_angle - dir_rad)
        bearing = wrap_angle_deg(bearing)

        pass_dx = df.loc[mask, "ball_land_x"].astype("float32") - df.loc[
            mask, "passer_x"
        ].astype("float32")
        pass_dy = df.loc[mask, "ball_land_y"].astype("float32") - df.loc[
            mask, "passer_y"
        ].astype("float32")
        pass_direction = np.rad2deg(np.arctan2(pass_dy, pass_dx))

        # Drop unused columns
        df.drop(columns=["passer_x", "passer_y"], inplace=True)

        # write back to df
        df.loc[mask, "passer_distance"] = dist
        df.loc[mask, "v_to_passer_alignment"] = align
        df.loc[mask, "v_to_passer_perp"] = perp
        df.loc[mask, "bearing_to_passer"] = bearing
        df.loc[mask, "pass_direction"] = pass_direction

        new_cols = [
            "passer_distance",
            "v_to_passer_alignment",
            "v_to_passer_perp",
            "bearing_to_passer",
            # "pass_direction",
        ]

        return df, new_cols

    def _create_curvature_features(self, df: pd.DataFrame):
        new_cols = []

        dx = df["ball_land_x"] - df["x"]
        dy = df["ball_land_y"] - df["y"]

        a_dir = np.deg2rad(df["dir"].fillna(0.0).values)

        # bearing signed
        bearing = np.arctan2(dy, dx)
        df["bearing_to_land_signed"] = np.rad2deg(
            np.arctan2(np.sin(bearing - a_dir), np.cos(bearing - a_dir))
        )

        # lateral offset (2D cross)
        ux, uy = np.cos(a_dir), np.sin(a_dir)
        df["land_lateral_offset"] = dy * ux - dx * uy

        # curvature
        dir_rad = np.deg2rad(df["dir"].fillna(0.0).values)
        curvature_signed = np.zeros(len(df), dtype="float32")

        df["_grp"] = pd.factorize(df[self.gcols].apply(tuple, axis=1))[0]
        grp_ids, grp_counts = np.unique(df["_grp"], return_counts=True)
        start_idx = 0
        for gid, cnt in tqdm(zip(grp_ids, grp_counts), total=len(grp_ids)):
            idx = slice(start_idx, start_idx + cnt)
            ddir = np.diff(dir_rad[idx], prepend=dir_rad[idx][0])
            # wrap [-pi, pi]
            ddir = (ddir + np.pi) % (2 * np.pi) - np.pi
            # curvature = delta_dir / (s * dt)
            s = df["s"].values[idx].astype("float32")
            curvature_signed[idx] = ddir / (s / Config.FPS + 1e-6)
            start_idx += cnt

        df["curvature_signed"] = curvature_signed
        df["curvature_abs"] = np.abs(curvature_signed)

        # Clear temporary columns
        df.drop(columns="_grp", inplace=True)

        new_cols = [
            "curvature_abs",
        ]

        return df, new_cols

    def _create_route_features(self, df: pd.DataFrame):
        # mask only players to predict
        mask = df["player_to_predict"] == 1
        sub = df[mask].copy()

        # Only use last 5 frames per player
        sub = sub.sort_values(self.gcols + ["frame_id"]).groupby(self.gcols).tail(5)

        # Compute diffs
        sub["dx"] = sub.groupby(self.gcols)["x"].diff()
        sub["dy"] = sub.groupby(self.gcols)["y"].diff()
        sub["ds"] = sub.groupby(self.gcols)["s"].diff()

        # Distance each step
        sub["step_dist"] = np.sqrt(sub["dx"] ** 2 + sub["dy"] ** 2)

        # angles -> second order angle change
        sub["angle"] = np.arctan2(sub["dy"], sub["dx"])
        sub["dangle"] = sub.groupby(self.gcols)["angle"].diff().abs()

        # Total distance & displacement
        feats = (
            sub.groupby(self.gcols)
            .agg(
                traj_total_dist=("step_dist", "sum"),
                start_x=("x", "first"),
                end_x=("x", "last"),
                start_y=("y", "first"),
                end_y=("y", "last"),
                speed_mean=("s", "mean"),
                speed_change=("s", lambda s: s.iloc[-1] - s.iloc[0]),
                traj_turn_ratio=("dangle", lambda a: (a > np.pi / 6).mean()),
            )
            .reset_index()
        )

        # displacement
        feats["traj_dx"] = feats["end_x"] - feats["start_x"]
        feats["traj_dy"] = feats["end_y"] - feats["start_y"]
        feats["traj_displacement"] = np.sqrt(
            feats["traj_dx"] ** 2 + feats["traj_dy"] ** 2
        )

        # straightness
        feats["traj_straightness"] = feats["traj_displacement"] / (
            feats["traj_total_dist"] + 0.1
        )

        # route depth / width / angle
        feats["traj_depth"] = feats["traj_dx"].abs()
        feats["traj_width"] = feats["traj_dy"].abs()
        feats["traj_direction_angle"] = np.arctan2(feats["traj_dy"], feats["traj_dx"])

        # Energy and momentum
        feats["traj_energy"] = feats["speed_mean"] ** 2 * feats["traj_total_dist"]
        feats["traj_momentum"] = feats["speed_mean"] * feats["traj_displacement"]

        turn = (
            sub.groupby(self.gcols)
            .agg(
                traj_max_turn=("dangle", "max"),
                traj_mean_turn=("dangle", "mean"),
            )
            .reset_index()
        )

        # Merge angle features
        feats = feats.merge(turn, on=self.gcols, how="left")

        feat_cols = [
            "speed_mean",
            "speed_change",
            # "traj_turn_ratio",
            "traj_straightness",
            "traj_depth",
            "traj_width",
            "traj_direction_angle",
            # "traj_energy",
            # "traj_momentum",
            "traj_max_turn",
            "traj_mean_turn",
        ]

        # merge back to df (only fill masked rows)
        df = df.merge(
            feats[self.gcols + feat_cols],
            on=self.gcols,
            how="left",
        )

        return df, feat_cols

    def _create_cooperation_features(self, df: pd.DataFrame, K: int = 3):
        new_cols = []

        info_cols = [
            "frame_id",
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "s",
            "a",
            "player_side",
        ]

        info_df = df[self.gcols + info_cols].copy()
        last_df = (
            info_df.sort_values(self.gcols + ["frame_id"])
            .groupby(self.gcols, as_index=False)
            .tail(1)
            .rename(columns={"frame_id": "frame_id_last"})
            .reset_index(drop=True)
        )

        nb_cols_map = {c: f"{c}_nb" for c in info_cols}

        info_df = last_df.merge(
            info_df.rename(columns=nb_cols_map),
            left_on=["game_id", "play_id", "frame_id_last"],
            right_on=["game_id", "play_id", "frame_id_nb"],
            how="left",
        )

        info_df = info_df[
            (info_df["nfl_id_nb"] != info_df["nfl_id"])
            & (info_df["player_side_nb"] == info_df["player_side"])
        ]

        dx = info_df["x_nb"] - info_df["x"]
        dy = info_df["y_nb"] - info_df["y"]
        info_df["dist"] = np.sqrt(dx**2 + dy**2)

        info_df = info_df[np.isfinite(info_df["dist"]) & (info_df["dist"] > 1e-3)]
        info_df = info_df[info_df["dist"] <= getattr(Config, "RADIUS", 15.0)]

        info_df["rnk"] = info_df.groupby(self.gcols)["dist"].rank(method="first")
        info_df = info_df[info_df["rnk"] <= K]

        info_df["vx_diff"] = info_df["velocity_x_nb"] - info_df["velocity_x"]
        info_df["vy_diff"] = info_df["velocity_y_nb"] - info_df["velocity_y"]
        info_df["speed_diff"] = info_df["s_nb"] - info_df["s"]
        info_df["acc_diff"] = info_df["a_nb"] - info_df["a"]

        info_df["angle_self"] = np.arctan2(info_df["velocity_y"], info_df["velocity_x"])
        info_df["angle_nb"] = np.arctan2(
            info_df["velocity_y_nb"], info_df["velocity_x_nb"]
        )
        info_df["angle_diff"] = np.abs(info_df["angle_self"] - info_df["angle_nb"])
        info_df["angle_diff"] = np.where(
            info_df["angle_diff"] > np.pi,
            2 * np.pi - info_df["angle_diff"],
            info_df["angle_diff"],
        )
        info_df["heading_align"] = np.cos(info_df["angle_diff"])

        info_df["approaching_rate"] = (
            dx * info_df["vx_diff"] + dy * info_df["vy_diff"]
        ) / (info_df["dist"] + 1e-3)

        ag = (
            info_df.groupby(self.gcols)
            .agg(
                coop_nearest_dist=("dist", "min"),
                coop_speed_diff=("speed_diff", "mean"),
                coop_acc_diff=("acc_diff", "mean"),
                coop_heading_align=("heading_align", "mean"),
                coop_approaching=("approaching_rate", "mean"),
            )
            .reset_index()
        )

        new_cols = [
            "coop_nearest_dist",
            "coop_speed_diff",
            "coop_acc_diff",
            "coop_heading_align",
            "coop_approaching",
        ]
        for c in new_cols:
            ag[c] = ag[c].fillna(0.0)

        df = df.merge(ag[self.gcols + new_cols], on=self.gcols, how="left")
        return df, new_cols

    def _create_receiver_features(self, df: pd.DataFrame):
        """Almost the same as `self._create_passer_features`"""
        # Get (x, y) position of receiver
        receiver_df = (
            df[df["player_role"] == "Targeted Receiver"]
            .groupby(["game_id", "play_id", "frame_id"], as_index=False)[["x", "y"]]
            .first()
            .rename(columns={"x": "receiver_x", "y": "receiver_y"})
        )

        df = df.merge(
            receiver_df,
            on=["game_id", "play_id", "frame_id"],
            how="left",
            validate="many_to_one",
        )
        mask = df["player_to_predict"]

        dx = df.loc[mask, "x"].astype("float32") - df.loc[mask, "receiver_x"].astype(
            "float32"
        )
        dy = df.loc[mask, "y"].astype("float32") - df.loc[mask, "receiver_y"].astype(
            "float32"
        )

        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        ux, uy = dx / dist, dy / dist

        vx = df.loc[mask, "velocity_x"].astype("float32")
        vy = df.loc[mask, "velocity_y"].astype("float32")

        # Projection
        align = vx * ux + vy * uy
        perp = vx * (-uy) + vy * ux

        dir_rad = np.deg2rad(df.loc[mask, "dir"].fillna(0).astype("float32"))

        # bearing
        to_receiver_angle = np.arctan2(-dy, -dx)
        bearing = np.rad2deg(to_receiver_angle - dir_rad)
        bearing = wrap_angle_deg(bearing)

        # write back to df
        df.loc[mask, "receiver_distance"] = dist
        df.loc[mask, "v_to_receiver_alignment"] = align
        df.loc[mask, "v_to_receiver_perp"] = perp
        df.loc[mask, "bearing_to_receiver"] = bearing

        new_cols = [
            "receiver_distance",
            "v_to_receiver_alignment",
            "v_to_receiver_perp",
            "bearing_to_receiver",
        ]

        return df, new_cols

    def transform(self, df: pd.DataFrame):
        df = df.copy().sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
        # # Use index to accelerate groupby and merge operations
        # df.set_index(self.gcols, inplace=True, drop=False)
        df = self._create_basic_features(df)

        # TODO: Optimize for interactive=False
        for group_name in self.active_groups:
            if group_name in self.feature_creators:
                creator, interactive = self.feature_creators[group_name]
                start_time = time.time()
                df, new_cols = creator(df)
                elapsed = time.time() - start_time
                self.created_feature_cols.extend(new_cols)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] [+] Added '{group_name}' "
                    f"({len(new_cols)} cols) in {elapsed:.2f}s\n    Columns: {new_cols}"
                )
            else:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] [!] Unknown feature group: {group_name}"
                )

        df = df[df["player_to_predict"]]
        final_cols = sorted(set(self.created_feature_cols))
        print(f"\nTotal features created: {len(final_cols)}")
        return df, final_cols


# === Model & Loss ===
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.02, lam_smooth=0.01):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.lam_smooth = lam_smooth

    def forward(self, pred, target, mask):
        # base huber
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta),
        )

        # time decay
        if self.time_decay and self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device, dtype=pred.dtype)
            w = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * w
            mask = mask.unsqueeze(-1) * w

        main_loss = (huber * mask).sum() / (mask.sum() + 1e-8)

        # # velocity smooth
        # if self.lam_smooth and pred.size(1) > 2:
        #     d1 = pred[:, 1:] - pred[:, :-1]
        #     d2 = d1[:, 1:] - d1[:, :-1]
        #     m2 = mask[:, 2:]
        #     smooth = (d2 * d2) * m2
        #     smooth_loss = smooth.sum() / (m2.sum() + 1e-8)
        # else:
        #     smooth_loss = pred.new_tensor(0.0)

        return main_loss


# Removed legacy RotaryEmbedding/rope utilities (not used)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(x) + x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = Config.MAX_FUTURE_HORIZON
        self.hidden_dim = Config.HIDDEN_DIM
        self.n_heads = Config.N_HEADS
        self.n_layers = Config.N_LAYERS
        self.n_querys = Config.N_QUERYS

        # 1. Spatio
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)

        # 2. Positional encoding for time and player axes
        self.pos_time = nn.Parameter(
            torch.randn(1, Config.WINDOW_SIZE, self.hidden_dim)
        )
        self.pos_player = nn.Parameter(
            torch.randn(
                1,
                Config.MAX_PLAYER + (1 if getattr(Config, "ADD_BALL_TOKEN", False) else 0),
                self.hidden_dim,
            )
        )
        self.embed_dropout = nn.Dropout(dropout)

        # 3a. Transformer Encoder (legacy / flattened)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # 3b. Axial encoders
        time_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        player_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=self.n_layers)
        self.player_encoder = nn.TransformerEncoder(player_layer, num_layers=self.n_layers)

        # 4. Anchored pooling: select targeted player's last time token
        self.pool_ln = nn.LayerNorm(self.hidden_dim)

        # 5. è¾“å‡º Head
        self.head = ResidualMLP(
            input_dim=self.n_querys * self.hidden_dim,
            hidden_dim=Config.MLP_HIDDEN_DIM,
            output_dim=self.horizon * 2,
            num_layers=Config.N_RES_BLOCKS,
            dropout=0.2,
        )

    def forward(self, x: torch.Tensor, player_mask: torch.Tensor = None):
        """Forward accepts either [B, T, D] or [B, P, T, D].

        - If [B, T, D]: legacy single-player input.
        - If [B, P, T, D]: play-level input with explicit player axis.
        Anchored pooling uses the targeted player's last time step. We assume
        targeted player is ordered as index 0 in the player axis.
        """
        if x.dim() == 3:
            # Legacy path: [B, T, D]
            B, T, _ = x.shape
            x_embed = self.input_projection(x)  # [B, T, H]
            x_embed = x_embed + self.pos_time[:, :T, :]
            x_embed = self.embed_dropout(x_embed)

            # Encode
            h = self.transformer_encoder(x_embed)  # [B, T, H]
            h = self.pool_ln(h)

            # Anchor on last time step
            ctx = h[:, -1, :]  # [B, H]

            out = self.head(ctx)
            out = out.view(B, self.horizon, 2)
            out = torch.cumsum(out, dim=1)
            return out

        elif x.dim() == 4:
            # Play-level path: [B, P, T, D]
            B, P, T, D = x.shape
            x_proj = self.input_projection(x)  # [B, P, T, H]
            pos_t = self.pos_time[:, :T, :].unsqueeze(1)  # [1, 1, T, H]
            if getattr(Config, "USE_AXIAL_ATTENTION", False):
                # Stage 1: temporal encoding per player
                x_t = x_proj + pos_t
                x_t = self.embed_dropout(x_t)
                h_t = self.time_encoder(x_t.reshape(B * P, T, self.hidden_dim))  # [BP, T, H]
                h_t = h_t.reshape(B, P, T, self.hidden_dim)

                # Stage 2: player-axis encoding at last time step
                h_last = h_t[:, :, T - 1, :]  # [B, P, H]
                pos_p = self.pos_player[:, :P, :]  # [1, P, H]
                h_players = self.embed_dropout(h_last + pos_p)

                pad_players = None
                if player_mask is not None and player_mask.dim() == 2:
                    pad_players = (player_mask == 0)  # [B, P] bool
                h_players = self.player_encoder(h_players, src_key_padding_mask=pad_players)  # [B, P, H]
                h_players = self.pool_ln(h_players)

                # Multi-query pooling
                q_idxs = [0]
                # ball token assumed appended as last index when enabled
                if getattr(Config, "ADD_BALL_TOKEN", False) and P >= (Config.MAX_PLAYER + 1):
                    q_idxs.append(P - 1)
                if len(q_idxs) < self.n_querys and P > 1:
                    q_idxs.append(1)
                q_idxs = q_idxs[: self.n_querys]
                ctx = torch.cat([h_players[:, qi, :] for qi in q_idxs], dim=-1)  # [B, nq*H]
            else:
                # Fallback: flattened PT attention
                pos_p = self.pos_player[:, :P, :].unsqueeze(2)  # [1, P, 1, H]
                x_embed = self.embed_dropout(x_proj + pos_t + pos_p)
                x_tok = x_embed.reshape(B, P * T, self.hidden_dim)
                pad_mask = None
                if player_mask is not None:
                    if player_mask.dim() == 2:
                        pm = player_mask.unsqueeze(-1).expand(B, P, T)
                    else:
                        pm = player_mask
                    pad_mask = (pm.reshape(B, P * T) == 0)
                h = self.transformer_encoder(x_tok, src_key_padding_mask=pad_mask)
                h = self.pool_ln(h)
                anchor_idx = T - 1
                ctx = h[:, anchor_idx, :]

            out = self.head(ctx)
            out = out.view(B, self.horizon, 2)
            out = torch.cumsum(out, dim=1)
            return out
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


# === Training & Evaluation ===
def train_model_stt(
    X_train,
    y_train_dx,
    y_train_dy,
    X_val,
    y_val_dx,
    y_val_dy,
    input_dim,
    pm_train=None,
    pm_val=None,
):
    device = Config.DEVICE
    print(f"[Device] Training on {device}")

    # Construct train/val dataset
    train_batches = []
    for i in range(0, len(X_train), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets_stt(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            Config.MAX_FUTURE_HORIZON,
        )
        if pm_train is not None:
            pm = torch.tensor(np.stack(pm_train[i:end]).astype(np.float32))
        else:
            pm = None
        train_batches.append((bx, by, bm, pm))

    val_batches = []
    for i in range(0, len(X_val), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets_stt(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            Config.MAX_FUTURE_HORIZON,
        )
        if pm_val is not None:
            pm = torch.tensor(np.stack(pm_val[i:end]).astype(np.float32))
        else:
            pm = None
        val_batches.append((bx, by, bm, pm))

    # Define model, criterion, optimizer, scheduler
    model = STTransformer(
        input_dim=input_dim,
    ).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5
    )
    # total_steps = Config.EPOCHS * len(train_batches)
    # warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_loss, best_state, bad = float("inf"), None, 0
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm, pm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pm = pm.to(device) if pm is not None else None
            pred = model(bx, player_mask=pm)
            loss = criterion(pred, by, bm)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        # Accumulate squared error for RMSE calculation across all validation batches
        se_sum = 0.0
        denom_sum = 0.0
        with torch.no_grad():
            for bx, by, bm, pm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pm = pm.to(device) if pm is not None else None
                pred = model(bx, player_mask=pm)
                val_losses.append(criterion(pred, by, bm).item())
                # Compute RMSE components on this batch
                pdx, pdy = pred[..., 0], pred[..., 1]
                ydx, ydy = by[..., 0], by[..., 1]
                mask = bm
                se_batch = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * mask
                se_sum += float(se_batch.sum().item())
                denom_sum += float(mask.sum().item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)

        # Compute epoch RMSE over validation set
        rmse_val = float(np.sqrt(se_sum / (2.0 * (denom_sum + 1e-8)))) if denom_sum > 0 else float('nan')

        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(
            f"  Epoch {epoch:>3}: train={train_loss:.4f}, val={val_loss:.4f}, rmse={rmse_val:.4f}, "
            f"Time_elapsed={minutes:>2}min {seconds:>2}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= Config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


def compute_val_rmse_stt(model, X_val_sc, ydx_list, ydy_list, horizon, device, pm_val=None):
    """Compute RMSE over validation sequences for STTransformer outputs.

    Parameters
    - model: trained STTransformer
    - X_val_sc: list or np.array of standardized input sequences
    - ydx_list, ydy_list: lists of target dx/dy arrays per sequence
    - horizon: prediction horizon length
    - device: torch device
    """
    # stack list -> np.array
    if isinstance(X_val_sc, list):
        X_val_sc = np.stack(X_val_sc).astype(np.float32)

    X_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    PM_t = None
    if pm_val is not None:
        PM_t = torch.tensor(np.stack(pm_val).astype(np.float32)).to(device)

    with torch.no_grad():
        predict = model(X_t, player_mask=PM_t).cpu().numpy()  # [B, H, 2]

    # targets & mask
    by, bm = prepare_targets_stt(ydx_list, ydy_list, horizon)
    if torch.is_tensor(by):
        by = by.numpy()
    if torch.is_tensor(bm):
        bm = bm.numpy()

    pdx, pdy = predict[..., 0], predict[..., 1]
    ydx, ydy = by[..., 0], by[..., 1]
    mask = bm

    # squared error
    se_sum2d = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * mask
    denom = mask.sum() + 1e-8

    return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))


def train_all_folds_stt(
    gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim, player_masks=None
):
    fold_rmses = []
    all_rmse = []
    cv_log = []

    for fold, (tr, va) in enumerate(gkf.split(sequences, y=None, groups=groups), 1):
        print(f"\n{'-'*60}\nFold {fold}/{Config.N_FOLDS} (seed {seed})\n{'-'*60}")

        X_tr = [sequences[i] for i in tr]
        X_va = [sequences[i] for i in va]
        PM_tr = [player_masks[i] for i in tr] if player_masks is not None else None
        PM_va = [player_masks[i] for i in va] if player_masks is not None else None
        y_tr_dx = [targets_dx[i] for i in tr]
        y_va_dx = [targets_dx[i] for i in va]
        y_tr_dy = [targets_dy[i] for i in tr]
        y_va_dy = [targets_dy[i] for i in va]

        scaler = StandardScaler()
        # Fit scaler over feature dimension using flattened player×time tokens when necessary
        if X_tr and np.array(X_tr[0]).ndim == 3:
            # [P, T, D] -> [-1, D]
            scaler.fit(np.vstack([s.reshape(-1, s.shape[-1]) for s in X_tr]))
        else:
            scaler.fit(np.vstack([s for s in X_tr]))

        if X_tr and np.array(X_tr[0]).ndim == 3:
            X_tr_sc = [scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape) for s in X_tr]
            X_va_sc = [scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape) for s in X_va]
        else:
            X_tr_sc = [scaler.transform(s) for s in X_tr]
            X_va_sc = [scaler.transform(s) for s in X_va]

        model, loss = train_model_stt(
            X_tr_sc,
            y_tr_dx,
            y_tr_dy,
            X_va_sc,
            y_va_dx,
            y_va_dy,
            input_dim,
            pm_train=PM_tr,
            pm_val=PM_va,
        )

        rmse = compute_val_rmse_stt(
            model,
            X_va_sc,
            [targets_dx[i] for i in va],
            [targets_dy[i] for i in va],
            Config.MAX_FUTURE_HORIZON,
            Config.DEVICE,
            pm_val=PM_va,
        )

        print(
            f"[VAL] seed {seed} fold {fold} at "
            f"Huber loss={loss:.5f} | "
            f"RMSE={rmse:.4f}"
        )

        fold_rmses.append(rmse)
        all_rmse.append(rmse)
        cv_log.append(
            {
                "seed": seed,
                "fold": fold,
                "rmse": rmse,
                "loss": float(loss),
            }
        )

        # Save model
        save_fold_artifacts_stt(
            seed=seed,
            fold=fold,
            scaler=scaler,
            model=model,
            base_dir=Config.SAVE_DIR,
        )

    print(
        f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
        f"mean={float(np.mean(fold_rmses)):.4f} yards"
    )

    return all_rmse, cv_log

def predict_sst(model, scaler, X_test_raw, device, pm_test=None):
    model.eval()
    outs_dx, outs_dy = [], []

    # Support both [T, D] and [P, T, D]
    if X_test_raw and np.array(X_test_raw[0]).ndim == 3:
        base = np.stack([
            scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
            for s in X_test_raw
        ]).astype(np.float32)
    else:
        base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)
    pm_t = None
    if pm_test is not None:
        pm_t = torch.tensor(np.stack(pm_test).astype(np.float32), device=device)

    with torch.no_grad():
        output = model(xt, player_mask=pm_t)

        dx = output[:, :, 0]
        dy = output[:, :, 1]

    outs_dx.append(dx.detach().cpu().numpy())
    outs_dy.append(dy.detach().cpu().numpy())

    return np.mean(outs_dx, axis=0), np.mean(outs_dy, axis=0)

# === Data Preparation ===
def _canonicalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("game_id", "play_id", "nfl_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Handle missing keys
    df = df.dropna(subset=["game_id", "play_id", "nfl_id"])
    # Convert to int64?
    df["game_id"] = df["game_id"].astype("int64")
    df["play_id"] = df["play_id"].astype("int64")
    df["nfl_id"] = df["nfl_id"].astype("int64")
    return df


def _process_group_batch(
    batch_keys: list,
    grouped_dict: dict,
    feature_cols: list,
    target_rows: pd.DataFrame,
    idx_x: int,
    idx_y: int,
    dir_map: pd.DataFrame,
    queue: Queue,
):
    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []
    for key in batch_keys:
        gid, pid, nid = key
        group_df = grouped_dict.get(key)
        if group_df is None:
            continue

        # Build input window
        input_window = group_df.tail(Config.WINDOW_SIZE)
        if len(input_window) < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - len(input_window)
            pad_df = pd.DataFrame(
                np.nan, index=range(pad_len), columns=input_window.columns
            )
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        input_window = input_window.fillna(input_window.mean(numeric_only=True))
        seq = input_window[feature_cols].to_numpy(dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        sequences.append(seq)

        # Training targets
        if Config.TRAIN:
            out_grp: pd.DataFrame = target_rows[
                (target_rows["game_id"] == gid)
                & (target_rows["play_id"] == pid)
                & (target_rows["nfl_id"] == nid)
            ].sort_values("frame_id")
            if len(out_grp) == 0:
                sequences.pop()
                continue
            dx = out_grp["x"].to_numpy(np.float32) - seq[-1, idx_x]
            dy = out_grp["y"].to_numpy(np.float32) - seq[-1, idx_y]
            fids = out_grp["frame_id"].to_numpy(np.int32)
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_fids.append(fids)

        play_dir_val = dir_map.loc[(gid, pid)]
        seq_meta.append(
            {
                "game_id": gid,
                "play_id": pid,
                "nfl_id": nid,
                "frame_id": int(input_window.iloc[-1]["frame_id"]),
                "play_direction": play_dir_val,
            }
        )

        if queue is not None:
            queue.put(1)

    return sequences, targets_dx, targets_dy, targets_fids, seq_meta


def prepare_sequences_with_advanced_features(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_groups: list = None,
):

    print(f"\n{'='*80}")
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES (UNIFIED FRAME)")
    print(f"{'='*80}")
    print(f"Window size: {Config.WINDOW_SIZE}")

    input_df = _canonicalize_key_dtypes(input_df)
    output_df = _canonicalize_key_dtypes(output_df)

    dir_map = build_play_direction_map(input_df)
    input_df = unify_left_direction_ipt(input_df)
    output_df = unify_left_direction_opt(output_df, dir_map)

    target_rows = output_df
    target_groups = output_df[["game_id", "play_id", "nfl_id"]].drop_duplicates()

    # Feature Engineering
    fe = FeatureEngineer(feature_groups)
    processed_df, base_feature_cols = fe.transform(input_df)

    # Build sequences
    start_time = time.time()
    grouped_dict = {
        (gid, pid, nid): g
        for (gid, pid, nid), g in processed_df.groupby(
            ["game_id", "play_id", "nfl_id"], sort=False
        )
    }

    # helpful indices
    idx_x = feature_cols.index("x")
    idx_y = feature_cols.index("y")

    # Spread group across cpus
    all_keys = [tuple(x) for x in target_groups.to_numpy()]
    batch_size = (len(all_keys) + Config.MAX_WORKER - 1) // Config.MAX_WORKER
    batches = [
        all_keys[i : i + batch_size] for i in range(0, len(all_keys), batch_size)
    ]

    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []

    if Config.TRAIN:
        manager = Manager()
        queue = manager.Queue()
        pbar = tqdm(total=len(all_keys), desc="Creating sequences (groups)")

        # Build sequences in parallel
        with ProcessPoolExecutor(max_workers=Config.MAX_WORKER) as ex:
            futures = [
                ex.submit(
                    _process_group_batch,
                    b,
                    grouped_dict,
                    feature_cols,
                    target_rows,
                    idx_x,
                    idx_y,
                    dir_map,
                    queue,
                )
                for b in batches
            ]
            finished = 0
            while finished < len(all_keys):
                queue.get()
                finished += 1
                pbar.update(1)

            # Wait for all task to complete
            for fut in as_completed(futures):
                seqs, dxs, dys, fids_list, metas = fut.result()
                sequences.extend(seqs)
                targets_dx.extend(dxs)
                targets_dy.extend(dys)
                targets_fids.extend(fids_list)
                seq_meta.extend(metas)

        pbar.close()

    else:
        # No multiprocessing when not training
        print("[INFO] Running in single-process mode")
        pbar = tqdm(total=len(all_keys), desc="Creating sequences (groups)")
        for key in all_keys:
            seqs, dxs, dys, fids_list, metas = _process_group_batch(
                [key],
                grouped_dict,
                feature_cols,
                target_rows,
                idx_x,
                idx_y,
                dir_map,
                None,
            )
            sequences.extend(seqs)
            seq_meta.extend(metas)
            pbar.update(1)
        pbar.close()
    end_time = time.time()
    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")
    print(f"Time to build sequences: {end_time - start_time:.2f} seconds")

    if Config.TRAIN:
        return (
            sequences,
            targets_dx,
            targets_dy,
            targets_fids,
            seq_meta,
            feature_cols,
        )
    return sequences, seq_meta, feature_cols


def _order_players_for_play(play_df: pd.DataFrame, target_nfl_id: int, end_frame: int) -> list:
    """Order players so that target comes first, followed by nearest players at end_frame."""
    # last positions per player at end_frame
    last = (
        play_df[play_df["frame_id"] == end_frame]
        .groupby("nfl_id", as_index=False)[["x", "y", "player_side"]]
        .first()
    )
    # target first
    if target_nfl_id not in set(last["nfl_id"].tolist()):
        # fallback to first in play_df
        target_nfl_id = int(play_df["nfl_id"].iloc[0])
    tx = float(last.loc[last["nfl_id"] == target_nfl_id, "x"].values[0])
    ty = float(last.loc[last["nfl_id"] == target_nfl_id, "y"].values[0])
    last["dist2target"] = np.hypot(last["x"] - tx, last["y"] - ty)
    other_ids = last[last["nfl_id"] != target_nfl_id].sort_values("dist2target")[
        "nfl_id"
    ].tolist()
    ordered = [target_nfl_id] + other_ids
    return ordered


def prepare_sequences_play_level(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_groups: list = None,
):
    """Build play-level sequences with explicit player axis: [P, T, D].

    Returns (when training):
    - sequences_4d: list of [P, T, D]
    - targets_dx, targets_dy: lists of target receiver dx/dy arrays
    - targets_fids: lists of frame_ids for targets
    - seq_meta: per-play metadata
    - feature_cols: list of feature names (same as FeatureEngineer output)
    - player_masks: list of [P] masks indicating valid players
    """
    print(f"\n{'='*80}")
    print(f"PREPARING PLAY-LEVEL SEQUENCES (PLAYER × TIME × FEATURES)")
    print(f"{'='*80}")
    print(f"Window size: {Config.WINDOW_SIZE} | Max players: {Config.MAX_PLAYER}")

    input_df = _canonicalize_key_dtypes(input_df)
    output_df = _canonicalize_key_dtypes(output_df)

    dir_map = build_play_direction_map(input_df)
    input_df = unify_left_direction_ipt(input_df)
    output_df = unify_left_direction_opt(output_df, dir_map)

    # Feature Engineering
    fe = FeatureEngineer(feature_groups)
    processed_df, base_feature_cols = fe.transform(input_df)

    # helpful indices for x,y
    idx_x = base_feature_cols.index("x")
    idx_y = base_feature_cols.index("y")

    sequences, player_masks, targets_dx, targets_dy, targets_fids, seq_meta = (
        [], [], [], [], [], []
    )

    # Iterate per (game_id, play_id)
    for (gid, pid), play_df in processed_df.groupby(["game_id", "play_id"], sort=False):
        play_df = play_df.sort_values(["nfl_id", "frame_id"]).reset_index(drop=True)
        end_frame = int(play_df["frame_id"].max())
        start_frame = max(end_frame - Config.WINDOW_SIZE + 1, int(play_df["frame_id"].min()))

        # Determine targeted receiver
        tr_ids = (
            play_df.loc[play_df["player_role"] == "Targeted Receiver", "nfl_id"].unique()
        )
        if len(tr_ids) == 0:
            # skip plays without targeted receiver
            continue
        target_nfl_id = int(tr_ids[0])

        # Order players: target first, then nearest at end_frame
        ordered_ids = _order_players_for_play(play_df, target_nfl_id, end_frame)
        selected_ids = ordered_ids[: Config.MAX_PLAYER]

        # Build per-player window and pad
        P = Config.MAX_PLAYER
        T = Config.WINDOW_SIZE
        D_base = len(base_feature_cols)
        play_tensor = np.zeros((P, T, D_base), dtype=np.float32)
        player_mask = np.zeros((P,), dtype=np.float32)

        # For reference: target last x,y for target alignment of targets
        ref_x, ref_y = None, None

        for p_idx in range(P):
            if p_idx >= len(selected_ids):
                # padded player
                continue
            nid = selected_ids[p_idx]
            sub = play_df[(play_df["nfl_id"] == nid) & (play_df["frame_id"].between(start_frame, end_frame))]
            # If fewer frames, pad at front
            if len(sub) < T:
                pad_len = T - len(sub)
                pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=sub.columns)
                sub = pd.concat([pad_df, sub], ignore_index=True)

            sub = sub.fillna(sub.mean(numeric_only=True))
            seq = sub[base_feature_cols].to_numpy(dtype=np.float32)
            seq = np.nan_to_num(seq, nan=0.0)

            play_tensor[p_idx] = seq[-T:]  # ensure exact window length
            player_mask[p_idx] = 1.0

            if nid == target_nfl_id:
                ref_x = float(seq[-1, idx_x])
                ref_y = float(seq[-1, idx_y])

        if ref_x is None or ref_y is None:
            # if target not found in window, skip
            continue

        # Targets for targeted receiver
        out_grp = (
            output_df[(output_df["game_id"] == gid) & (output_df["play_id"] == pid) & (output_df["nfl_id"] == target_nfl_id)]
            .sort_values("frame_id")
        )
        if len(out_grp) == 0:
            # skip if no target rows
            continue
        dx = out_grp["x"].to_numpy(np.float32) - np.float32(ref_x)
        dy = out_grp["y"].to_numpy(np.float32) - np.float32(ref_y)
        fids = out_grp["frame_id"].to_numpy(np.int32)

        # --- Step 4 (optional): time-varying neighbor features per frame ---
        try:
            # Player sides (ally/opp) inferred from last frame
            sides_last = (
                play_df[play_df["frame_id"] == end_frame]
                .groupby("nfl_id", as_index=False)["player_side"]
                .first()
            )
            side_map = {int(r["nfl_id"]): str(r["player_side"]) for _, r in sides_last.iterrows()}
            sides = [side_map.get(int(nid), "Offense") for nid in selected_ids]  # default Offense

            # Positions per frame
            X = play_tensor[:, :, idx_x]  # [P, T]
            Y = play_tensor[:, :, idx_y]  # [P, T]
            tv_pressure = np.zeros((P, T), dtype=np.float32)
            tv_ally_density = np.zeros((P, T), dtype=np.float32)
            tv_oppn_density = np.zeros((P, T), dtype=np.float32)
            tv_density_ratio = np.zeros((P, T), dtype=np.float32)
            tv_dist_min = np.zeros((P, T), dtype=np.float32)

            R = float(getattr(Config, "RADIUS", 30.0))
            for t in range(T):
                # collect valid players at time t
                valid_idx = [i for i in range(P) if player_mask[i] > 0.5]
                if len(valid_idx) <= 1:
                    continue
                coords = np.stack([X[valid_idx, t], Y[valid_idx, t]], axis=-1)  # [V, 2]
                for loc_pos, i in enumerate(valid_idx):
                    xi, yi = coords[loc_pos]
                    # distances to all others
                    diffs = coords - np.array([xi, yi], dtype=np.float32)
                    dists = np.sqrt((diffs ** 2).sum(axis=-1) + 1e-8)
                    dists[loc_pos] = np.inf  # exclude self
                    tv_dist_min[i, t] = np.min(dists)

                    # ally/opp masks based on player_side
                    si = sides[loc_pos]
                    ally_mask = np.array([1.0 if sides[j] == si else 0.0 for j in range(len(valid_idx))], dtype=np.float32)
                    ally_mask[loc_pos] = 0.0
                    opp_mask = 1.0 - ally_mask

                    # density within radius R (count / area)
                    ally_count = float(np.sum((dists <= R) * ally_mask))
                    opp_count = float(np.sum((dists <= R) * opp_mask))
                    area = np.pi * (R ** 2)
                    tv_ally_density[i, t] = ally_count / (area + 1e-6)
                    tv_oppn_density[i, t] = opp_count / (area + 1e-6)
                    tv_density_ratio[i, t] = tv_ally_density[i, t] / (tv_oppn_density[i, t] + 1e-6)

                    # pressure as inverse of nearest opponent distance
                    opp_dists = dists[opp_mask > 0.5]
                    if opp_dists.size == 0:
                        opp_near = np.inf
                    else:
                        opp_near = float(np.min(opp_dists))
                    tv_pressure[i, t] = 1.0 / max(opp_near, 0.5)

            # concatenate new features to play_tensor along feature dimension
            new_feats = np.stack(
                [tv_pressure, tv_ally_density, tv_oppn_density, tv_density_ratio, tv_dist_min],
                axis=-1,
            )  # [P, T, 5]
            play_tensor = np.concatenate([play_tensor, new_feats], axis=-1)
        except Exception as e:
            # Keep shapes consistent across plays by appending zeros if computation fails
            print(f"[WARN] time-varying neighbor features failed for play ({gid}, {pid}): {e}")
            zeros_feats = np.zeros((P, T, 5), dtype=np.float32)
            play_tensor = np.concatenate([play_tensor, zeros_feats], axis=-1)

        # Optionally append a shared ball token as an extra player
        if getattr(Config, "ADD_BALL_TOKEN", False):
            try:
                bx_val = float(play_df["ball_land_x"].dropna().iloc[-1])
                by_val = float(play_df["ball_land_y"].dropna().iloc[-1])
            except Exception:
                bx_val = float(play_df["x"].iloc[-1])
                by_val = float(play_df["y"].iloc[-1])

            D_total = int(play_tensor.shape[-1])
            ball_seq = np.zeros((T, D_total), dtype=np.float32)

            # Write into base feature indices for x/y and ball landing
            bx_i = base_feature_cols.index("ball_land_x") if "ball_land_x" in base_feature_cols else None
            by_i = base_feature_cols.index("ball_land_y") if "ball_land_y" in base_feature_cols else None
            x_i = idx_x
            y_i = idx_y
            ball_seq[:, x_i] = bx_val
            ball_seq[:, y_i] = by_val
            if bx_i is not None:
                ball_seq[:, bx_i] = bx_val
            if by_i is not None:
                ball_seq[:, by_i] = by_val

            play_tensor = np.concatenate([play_tensor, ball_seq[np.newaxis, :, :]], axis=0)
            player_mask = np.concatenate([player_mask, np.array([1.0], dtype=np.float32)], axis=0)

        sequences.append(play_tensor)
        player_masks.append(player_mask)
        targets_dx.append(dx)
        targets_dy.append(dy)
        targets_fids.append(fids)
        seq_meta.append({"game_id": gid, "play_id": pid, "target_nfl_id": target_nfl_id})

    # Final feature columns include base + time-varying neighbor features appended to tensor
    extra_cols = [
        "tv_pressure",
        "tv_ally_density",
        "tv_oppn_density",
        "tv_density_ratio",
        "tv_dist_min",
    ]
    feature_cols_total = base_feature_cols + extra_cols

    print(f"Created {len(sequences)} play-level sequences with shape [P={Config.MAX_PLAYER}, T={Config.WINDOW_SIZE}, D={len(feature_cols_total)}]")

    if Config.TRAIN:
        return (
            sequences,
            targets_dx,
            targets_dy,
            targets_fids,
            seq_meta,
            feature_cols_total,
            player_masks,
        )
    return sequences, seq_meta, feature_cols_total, player_masks


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
