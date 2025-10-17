import os, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool as MP, cpu_count
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool as CatPool
import xgboost as xgb
import lightgbm as lgb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# Official Kaggle scoring function (copied from scoring.py)
class ParticipantVisibleError(Exception):
    pass

TARGET = ['x', 'y']

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute RMSE for NFL competition.
    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'id': unique identifier for each (game_id, play_id, nfl_id, frame_id)
      - Column 'x'
      - Column 'y'
    """
    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(f"Solution file missing required column: '{row_id_column_name}'")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission file missing required column: '{row_id_column_name}'")

    missing_in_solution = set(TARGET) - set(solution.columns)
    missing_in_submission = set(TARGET) - set(submission.columns)

    if missing_in_solution:
        raise ParticipantVisibleError(f'Solution file missing required columns: {missing_in_solution}')
    if missing_in_submission:
        raise ParticipantVisibleError(f'Submission file missing required columns: {missing_in_submission}')

    submission = submission[['id'] + TARGET]
    merged_df = pd.merge(solution, submission, on=row_id_column_name, suffixes=('_true', '_pred'))
    rmse = np.sqrt(
        0.5 * (
            mean_squared_error(merged_df['x_true'], merged_df['x_pred']) 
            + mean_squared_error(merged_df['y_true'], merged_df['y_pred'])
        )
    )
    return float(rmse)

class NFLConfig:
    def __init__(self):
        self.BASEDIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction")
        self.SAVE_DIR = Path("/kaggle/working")
        
        self.N_WEEKS = 18
        self.N_FOLDS = 5
        
        # Model parameters
        self.ITERATIONS = 20000
        self.LR = 0.08
        self.DEPTH = 8
        self.L2 = 3.0
        self.EARLY = 500
        
        self.SEED = 42
        self.USE_GPU = bool(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        self.USE_GROUP_KFOLD = True

        # GNN-lite parameters
        self.K_NEIGH = 6
        self.RADIUS = 30.0
        self.TAU = 8.0

        # Ensemble settings (ported from main_mps.py)
        # Global weights for tree-based ensemble contributions
        self.ENSEMBLE_WEIGHTS = {
            'catboost': 0.5, 
            'xgboost': 0.1, 
            'lightgbm': 0.1,
            'neural_net': 0.3
        }
        # Whether to use stacking meta-learners (not enabled by default here)
        self.USE_STACKING = False
        # Whether to use residual learning ensemble (CatBoost primary + XGB/LGB residuals)
        self.USE_RESIDUAL_LEARNING = True

        # Optional hyperparameters for additional GBMs
        self.XGB_N_ESTIMATORS = 1500
        self.LGB_N_ESTIMATORS = 1500
        
        # Neural Network parameters
        self.NN_BATCH_SIZE = 256
        self.NN_EPOCHS = 200
        self.NN_PATIENCE = 30
        self.NN_LEARNING_RATE = 1e-3
        self.NN_WINDOW_SIZE = 8
        self.NN_HIDDEN_DIM = 128
        
        # Device configuration for CUDA
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")

class DataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_week(self, week_num: int):
        fin = self.config.BASEDIR / f"train/input_2023_w{week_num:02d}.csv"
        fout = self.config.BASEDIR / f"train/output_2023_w{week_num:02d}.csv"
        return pd.read_csv(fin), pd.read_csv(fout)
    
    def load_all_train(self):
        print("Loading training data...")
        with MP(min(cpu_count(), 18)) as pool:
            res = list(
                tqdm(
                    pool.imap(self.load_week, range(1, self.config.N_WEEKS+1)), 
                    total=self.config.N_WEEKS
                )
            )
        tr_in  = pd.concat([r[0] for r in res], ignore_index=True)
        tr_out = pd.concat([r[1] for r in res], ignore_index=True)
        print(f"Train input:  {tr_in.shape}")
        print(f"Train output: {tr_out.shape}")
        return tr_in, tr_out
    
    def load_test_data(self):
        te_in = pd.read_csv(self.config.BASEDIR / "test_input.csv")
        te_tpl = pd.read_csv(self.config.BASEDIR / "test.csv")
        return te_in, te_tpl

class FeatureEngineer:
    # Imperial conversion
    @staticmethod
    def to_inches(h):
        try:
            a, b = str(h).split("-")
            return float(a)*12.0 + float(b)
        except Exception:
            return np.nan
    
    # Physics/geometry with CORRECT NFL angle convention (dir=0 -> +y)
    @staticmethod
    def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Height/BMI
        df["height_inches"] = df["player_height"].map(FeatureEngineer.to_inches)
        df["bmi"] = (df["player_weight"] / (df["height_inches"]**2)) * 703.0

        # Heading unit (dir=0° points to +y)
        dir_rad = np.radians(df["dir"].fillna(0.0))
        df["heading_x"] = np.sin(dir_rad)
        df["heading_y"] = np.cos(dir_rad)

        # Velocity/Acceleration (correct axes)
        s = df["s"].fillna(0.0)
        a = df["a"].fillna(0.0)
        df["velocity_x"] = s * df["heading_x"]
        df["velocity_y"] = s * df["heading_y"]
        df["acceleration_x"] = a * df["heading_x"]
        df["acceleration_y"] = a * df["heading_y"]

        # Target (ball landing) geometry
        dx = df["ball_land_x"] - df["x"]
        dy = df["ball_land_y"] - df["y"]
        dist = np.sqrt(dx**2 + dy**2)
        df["dist_to_ball"] = dist
        df["angle_to_ball"] = np.arctan2(dy, dx)
        bux = dx / (dist + 1e-6)
        buy = dy / (dist + 1e-6)

        # Velocity toward ball & alignment
        df["velocity_toward_ball"] = df["velocity_x"]*bux + df["velocity_y"]*buy
        df["velocity_alignment"]   = df["heading_x"]*bux + df["heading_y"]*buy

        # Other physics
        df["speed_squared"]   = s**2
        df["accel_magnitude"] = np.sqrt(df["acceleration_x"]**2 + df["acceleration_y"]**2)
        w = df["player_weight"].fillna(0.0)
        df["momentum_x"] = w * df["velocity_x"]
        df["momentum_y"] = w * df["velocity_y"]
        df["kinetic_energy"] = 0.5 * w * df["speed_squared"]

        # Roles / side
        df["role_targeted_receiver"] = (df["player_role"] == "Targeted Receiver").astype(int)
        df["role_defensive_coverage"] = (df["player_role"] == "Defensive Coverage").astype(int)
        df["role_passer"] = (df["player_role"] == "Passer").astype(int)
        df["side_offense"] = (df["player_side"] == "Offense").astype(int)

        return df
    
    # Sequence features (lags & rolling)
    @staticmethod
    def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["game_id","play_id","nfl_id","frame_id"])
        gcols = ["game_id","play_id","nfl_id"]

        for lag in [1,2,3,4,5]:
            for c in ["x","y","velocity_x","velocity_y","s","a"]:
                if c in df.columns:
                    df[f"{c}_lag{lag}"] = df.groupby(gcols)[c].shift(lag)

        for win in [3,5]:
            for c in ["x","y","velocity_x","velocity_y","s"]:
                if c in df.columns:
                    df[f"{c}_rolling_mean_{win}"] = (
                        df.groupby(gcols)[c].rolling(win, min_periods=1).mean()
                          .reset_index(level=[0,1,2], drop=True)
                    )
                    df[f"{c}_rolling_std_{win}"] = (
                        df.groupby(gcols)[c].rolling(win, min_periods=1).std()
                          .reset_index(level=[0,1,2], drop=True)
                    )

        for c in ["velocity_x","velocity_y"]:
            if c in df.columns:
                df[f"{c}_delta"] = df.groupby(gcols)[c].diff()

        return df

    # New method for ensemble-specific feature engineering
    @staticmethod
    def add_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        gcols = ["game_id","play_id","nfl_id"]

        # Interaction features between position and velocity
        if all(c in df.columns for c in ["x","y","velocity_x","velocity_y"]):
            df["pos_vel_interaction_x"] = df["x"] * df["velocity_x"]
            df["pos_vel_interaction_y"] = df["y"] * df["velocity_y"]

        # Distance-based features relative to field
        if all(c in df.columns for c in ["x","y"]):
            df["distance_from_center"] = np.sqrt((df["x"] - 60)**2 + (df["y"] - 26.65)**2)
            df["distance_from_sideline"] = np.minimum(df["y"], 53.3 - df["y"])
            df["distance_from_endzone"] = np.minimum(df["x"], 120 - df["x"]) 

        # Formation-based statistics per team side
        for stat in ["mean", "std", "min", "max"]:
            if "player_side" in df.columns and all(c in df.columns for c in ["x","y","s"]):
                df[f"team_x_{stat}"] = df.groupby(["game_id", "play_id", "player_side"])['x'].transform(stat)
                df[f"team_y_{stat}"] = df.groupby(["game_id", "play_id", "player_side"])['y'].transform(stat)
                df[f"team_speed_{stat}"] = df.groupby(["game_id", "play_id", "player_side"])['s'].transform(stat)

        # Relative to team centroid
        if all(c in df.columns for c in ["x","y","s"]):
            df["rel_x_to_team"] = df["x"] - df.get("team_x_mean", 0.0)
            df["rel_y_to_team"] = df["y"] - df.get("team_y_mean", 0.0)
            df["rel_speed_to_team"] = df["s"] - df.get("team_speed_mean", 0.0)

        # Time-based cyclical features
        if "frame_id" in df.columns:
            df["frame_sin"] = np.sin(2 * np.pi * df["frame_id"] / 10)
            df["frame_cos"] = np.cos(2 * np.pi * df["frame_id"] / 10)

        # Velocity angle features
        if all(c in df.columns for c in ["velocity_x","velocity_y"]):
            df["velocity_angle"] = np.arctan2(df["velocity_y"], df["velocity_x"]) 
            df["velocity_angle_sin"] = np.sin(df["velocity_angle"]) 
            df["velocity_angle_cos"] = np.cos(df["velocity_angle"]) 

        # Acceleration alignment with velocity
        if all(c in df.columns for c in ["acceleration_x","acceleration_y","velocity_x","velocity_y"]):
            velocity_mag = np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2) + 1e-6
            accel_mag = np.sqrt(df["acceleration_x"]**2 + df["acceleration_y"]**2) + 1e-6
            df["accel_vel_alignment"] = (
                (df["acceleration_x"] * df["velocity_x"] + df["acceleration_y"] * df["velocity_y"]) /
                (velocity_mag * accel_mag)
            )

        # Route running patterns (for receivers)
        if "direction_change" in df.columns and "s" in df.columns:
            df["route_sharpness"] = (
                df.groupby(gcols)["direction_change"].rolling(3, min_periods=1).std().reset_index(level=[0,1,2], drop=True)
            )
            df["route_consistency"] = (
                df.groupby(gcols)["s"].rolling(5, min_periods=1).std().reset_index(level=[0,1,2], drop=True)
            )

        return df

# GNN-lite neighbor embedding computations
class GNNLiteProcessor:
    def __init__(self, config):
        self.config = config
    
    def compute_neighbor_embeddings(self, input_df: pd.DataFrame) -> pd.DataFrame:
        cols_needed = ["game_id","play_id","nfl_id","frame_id","x","y",
                       "velocity_x","velocity_y","player_side"]
        src = input_df[cols_needed].copy()

        last = (src.sort_values(["game_id","play_id","nfl_id","frame_id"])
                   .groupby(["game_id","play_id","nfl_id"], as_index=False)
                   .tail(1)
                   .rename(columns={"frame_id":"last_frame_id"})
                   .reset_index(drop=True))

        # join neighbors at the ego's last_frame_id
        tmp = last.merge(
            src.rename(columns={
                "frame_id":"nb_frame_id", "nfl_id":"nfl_id_nb",
                "x":"x_nb", "y":"y_nb",
                "velocity_x":"vx_nb", "velocity_y":"vy_nb",
                "player_side":"player_side_nb"
            }),
            left_on=["game_id","play_id","last_frame_id"],
            right_on=["game_id","play_id","nb_frame_id"],
            how="left",
        )

        # drop self
        tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]

        # relative vectors
        tmp["dx"]  = tmp["x_nb"] - tmp["x"]
        tmp["dy"]  = tmp["y_nb"] - tmp["y"]
        tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
        tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
        tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)

        tmp = tmp[np.isfinite(tmp["dist"])]
        tmp = tmp[tmp["dist"] > 1e-6]
        if self.config.RADIUS is not None:
            tmp = tmp[tmp["dist"] <= self.config.RADIUS]

        # ally / opp flag
        tmp["is_ally"] = (tmp["player_side_nb"].fillna("") == tmp["player_side"].fillna("")).astype(np.float32)

        # rank by distance (keep top-K)
        keys = ["game_id","play_id","nfl_id"]
        tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
        if self.config.K_NEIGH is not None:
            tmp = tmp[tmp["rnk"] <= float(self.config.K_NEIGH)]

        # attention weights: softmax(-dist/tau) within group
        tmp["w"] = np.exp(-tmp["dist"] / float(self.config.TAU))
        sum_w = tmp.groupby(keys)["w"].transform("sum")
        tmp["wn"] = np.where(sum_w > 0, tmp["w"]/sum_w, 0.0)

        tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
        tmp["wn_opp"]  = tmp["wn"] * (1.0 - tmp["is_ally"])

        # pre-multiply for group sums
        for col in ["dx","dy","dvx","dvy"]:
            tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
            tmp[f"{col}_opp_w"]  = tmp[col] * tmp["wn_opp"]

        tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
        tmp["dist_opp"]  = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)

        ag = tmp.groupby(keys).agg(
            gnn_ally_dx_mean = ("dx_ally_w", "sum"),
            gnn_ally_dy_mean = ("dy_ally_w", "sum"),
            gnn_ally_dvx_mean= ("dvx_ally_w","sum"),
            gnn_ally_dvy_mean= ("dvy_ally_w","sum"),
            gnn_opp_dx_mean  = ("dx_opp_w",  "sum"),
            gnn_opp_dy_mean  = ("dy_opp_w",  "sum"),
            gnn_opp_dvx_mean = ("dvx_opp_w", "sum"),
            gnn_opp_dvy_mean = ("dvy_opp_w", "sum"),
            gnn_ally_cnt     = ("is_ally",   "sum"),
            gnn_opp_cnt      = ("is_ally",   lambda s: float(len(s) - s.sum())),
            gnn_ally_dmin    = ("dist_ally", "min"),
            gnn_ally_dmean   = ("dist_ally", "mean"),
            gnn_opp_dmin     = ("dist_opp",  "min"),
            gnn_opp_dmean    = ("dist_opp",  "mean"),
        ).reset_index()

        # d1..d3 nearest (regardless of side)
        near = tmp.loc[tmp["rnk"]<=3, keys+["rnk","dist"]].copy()
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1:"gnn_d1",2:"gnn_d2",3:"gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")

        # safe fills
        for c in ["gnn_ally_dx_mean","gnn_ally_dy_mean","gnn_ally_dvx_mean","gnn_ally_dvy_mean",
                  "gnn_opp_dx_mean","gnn_opp_dy_mean","gnn_opp_dvx_mean","gnn_opp_dvy_mean"]:
            ag[c] = ag[c].fillna(0.0)
        for c in ["gnn_ally_cnt","gnn_opp_cnt"]:
            ag[c] = ag[c].fillna(0.0)
        for c in ["gnn_ally_dmin","gnn_opp_dmin","gnn_ally_dmean","gnn_opp_dmean","gnn_d1","gnn_d2","gnn_d3"]:
            ag[c] = ag[c].fillna(self.config.RADIUS if self.config.RADIUS is not None else 30.0)

        return ag

class TrainingDataBuilder:
    # Builds training data with physics baseline and residuals
    # Merge each future frame with the LAST observed stats for that (gid,pid,nfl) + Δt
    @staticmethod
    def create_training_rows(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            input_df.sort_values(["game_id","play_id","nfl_id","frame_id"])
                    .groupby(["game_id","play_id","nfl_id"], as_index=False)
                    .tail(1)
                    .reset_index(drop=True)
                    .rename(columns={"frame_id":"last_frame_id"})
        )

        out = output_df.copy()
        out = out.rename(columns={"x":"target_x","y":"target_y"})
        out["id"] = (
            out["game_id"].astype(str) + "_" +
            out["play_id"].astype(str) + "_" +
            out["nfl_id"].astype(str) + "_" +
            out["frame_id"].astype(str)
        )

        m = out.merge(
            agg,
            on=["game_id","play_id","nfl_id"],
            how="left",
            suffixes=("","_last")
        )

        m["delta_frames"] = (m["frame_id"] - m["last_frame_id"]).clip(lower=0).astype(float)
        m["delta_t"] = m["delta_frames"] / 10.0
        return m
    
    @staticmethod
    def physics_baseline(x_last, y_last, vx_last, vy_last, dt):
        px = x_last + vx_last * dt
        py = y_last + vy_last * dt
        px = np.clip(px, 0.0, 120.0)
        py = np.clip(py, 0.0, 53.3)
        return px, py
    
    @staticmethod
    def build_feature_list(train_df: pd.DataFrame):
        base = [
            "x","y","s","a","o","dir",
            "velocity_x","velocity_y",
            "acceleration_x","acceleration_y",
            "heading_x","heading_y",
            "player_weight","height_inches","bmi",
            "ball_land_x","ball_land_y",
            "dist_to_ball","angle_to_ball",
            "velocity_toward_ball","velocity_alignment",
            "speed_squared","accel_magnitude","momentum_x","momentum_y","kinetic_energy",
            "role_targeted_receiver","role_defensive_coverage","role_passer","side_offense",
            "delta_frames","delta_t",
            "frame_id",
            # GNN-lite
            "gnn_ally_dx_mean","gnn_ally_dy_mean","gnn_ally_dvx_mean","gnn_ally_dvy_mean",
            "gnn_opp_dx_mean","gnn_opp_dy_mean","gnn_opp_dvx_mean","gnn_opp_dvy_mean",
            "gnn_ally_cnt","gnn_opp_cnt",
            "gnn_ally_dmin","gnn_ally_dmean","gnn_opp_dmin","gnn_opp_dmean",
            "gnn_d1","gnn_d2","gnn_d3",
            # Ensemble diversity features
            "pos_vel_interaction_x","pos_vel_interaction_y",
            "distance_from_center","distance_from_sideline","distance_from_endzone",
            "team_x_mean","team_x_std","team_x_min","team_x_max",
            "team_y_mean","team_y_std","team_y_min","team_y_max",
            "team_speed_mean","team_speed_std","team_speed_min","team_speed_max",
            "rel_x_to_team","rel_y_to_team","rel_speed_to_team",
            "frame_sin","frame_cos",
            "velocity_angle","velocity_angle_sin","velocity_angle_cos",
            "accel_vel_alignment",
            "route_sharpness","route_consistency",
        ]
        for lag in [1,2,3,4,5]:
            for c in ["x","y","velocity_x","velocity_y","s","a"]:
                base.append(f"{c}_lag{lag}")
        for win in [3,5]:
            for c in ["x","y","velocity_x","velocity_y","s"]:
                base.append(f"{c}_rolling_mean_{win}")
                base.append(f"{c}_rolling_std_{win}")
        base += ["velocity_x_delta","velocity_y_delta"]

        feats = [c for c in base if c in train_df.columns]
        # include any extra 'gnn_' columns that exist
        feats = list(dict.fromkeys(feats + [c for c in train_df.columns if c.startswith("gnn_")]))
        return feats

class TemporalHuber(nn.Module):
    """Temporal Huber loss with frame-based weighting"""
    def __init__(self, delta=1.0, temporal_weight=0.1):
        super().__init__()
        self.delta = delta
        self.temporal_weight = temporal_weight
        
    def forward(self, pred, target, frame_weights=None):
        residual = torch.abs(pred - target)
        
        # Huber loss
        loss = torch.where(
            residual < self.delta,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        
        # Apply frame weights if provided
        if frame_weights is not None:
            loss = loss * frame_weights.unsqueeze(-1)
            
        return loss.mean()

class SeqModel(nn.Module):
    """GRU-based sequence model for player trajectory prediction"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature normalization
        self.norm = nn.LayerNorm(input_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)  # x, y coordinates
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.norm(x)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Use last timestep output
        last_output = gru_out[:, -1, :]
        
        # Prediction layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class NeuralNetworkTrainer:
    """Handles neural network training with sequence data"""
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
    def prepare_sequence_data(self, df, feature_cols, window_size=8):
        """Prepare sequence data for GRU training"""
        sequences = []
        targets = []
        
        # Group by game, play, player
        for (game_id, play_id, nfl_id), group in df.groupby(['game_id', 'play_id', 'nfl_id']):
            group = group.sort_values('frame_id')
            
            if len(group) < window_size:
                continue
                
            # Create sequences
            for i in range(len(group) - window_size + 1):
                seq_data = group.iloc[i:i+window_size][feature_cols].values
                target_data = group.iloc[i+window_size-1][['target_x', 'target_y']].values
                
                sequences.append(seq_data)
                targets.append(target_data)
        
        return np.array(sequences), np.array(targets)
    
    def train_neural_network_folds(self, X, yx, yy, ids_group=None, feature_cols=None):
        """Train neural network with cross-validation"""
        if feature_cols is None:
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            
        # Prepare data for neural network
        df_temp = pd.DataFrame(X, columns=feature_cols)
        df_temp['target_x'] = yx
        df_temp['target_y'] = yy
        df_temp['game_id'] = range(len(df_temp))  # Dummy grouping
        df_temp['play_id'] = 0
        df_temp['nfl_id'] = 0
        df_temp['frame_id'] = range(len(df_temp))
        
        # Create folds
        folds = []
        if self.config.USE_GROUP_KFOLD and ids_group is not None:
            gkf = GroupKFold(n_splits=self.config.N_FOLDS)
            for tr, va in gkf.split(X, groups=ids_group):
                folds.append((tr, va))
        else:
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
            folds = list(kf.split(X))
        
        models = []
        fold_rmse = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"Training Neural Network Fold {fold_idx + 1}/{self.config.N_FOLDS}")
            
            # Split data
            train_df = df_temp.iloc[train_idx]
            val_df = df_temp.iloc[val_idx]
            
            # Prepare sequences
            X_train_seq, y_train = self.prepare_sequence_data(train_df, feature_cols, self.config.NN_WINDOW_SIZE)
            X_val_seq, y_val = self.prepare_sequence_data(val_df, feature_cols, self.config.NN_WINDOW_SIZE)
            
            if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                print(f"Skipping fold {fold_idx} due to insufficient sequence data")
                continue
            
            # Create model
            model = SeqModel(
                input_dim=len(feature_cols),
                hidden_dim=self.config.NN_HIDDEN_DIM,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            
            # Train model
            trained_model = self._train_single_model(model, X_train_seq, y_train, X_val_seq, y_val)
            models.append(trained_model)
            
            # Evaluate
            with torch.no_grad():
                trained_model.eval()
                X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
                val_pred = trained_model(X_val_tensor).cpu().numpy()
                
                rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], val_pred[:, 0]))
                rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], val_pred[:, 1]))
                rmse = np.sqrt(0.5 * (rmse_x**2 + rmse_y**2))
                fold_rmse.append(rmse)
                print(f"Fold {fold_idx + 1} RMSE: {rmse:.4f}")
        
        return {
            'models': models,
            'fold_rmse': fold_rmse,
            'mean_rmse': np.mean(fold_rmse) if fold_rmse else float('inf')
        }
    
    def _train_single_model(self, model, X_train, y_train, X_val, y_val):
        """Train a single neural network model"""
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.NN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.NN_BATCH_SIZE, shuffle=False)
        
        # Loss and optimizer
        criterion = TemporalHuber(delta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.NN_LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.NN_EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.NN_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model

class CatBoostTrainer:
    # Handles CatBoost model training with cross-validation
    def __init__(self, config):
        self.config = config
    
    def train_catboost_folds(self, X, yx, yy, ids_group=None,
                            base_x=None, base_y=None, use_residual=False):
        folds = []
        if self.config.USE_GROUP_KFOLD and ids_group is not None:
            print("Using GroupKFold by (game_id,play_id,nfl_id).")
            gkf = GroupKFold(n_splits=self.config.N_FOLDS)
            for tr, va in gkf.split(X, groups=ids_group):
                folds.append((tr, va))
        else:
            print("Using plain KFold.")
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
            folds = list(kf.split(X))

        models_x, models_y, fold_rmse = [], [], []
        task_type = "GPU" if self.config.USE_GPU else "CPU"
        devices = "0:1" if self.config.USE_GPU else None

        for i, (tr, va) in enumerate(folds, 1):
            print(f"\nFold {i}/{self.config.N_FOLDS} — train {len(tr):,} | val {len(va):,}")

            Xtr, Xva = X[tr], X[va]
            yx_tr, yx_va = yx[tr], yx[va]
            yy_tr, yy_va = yy[tr], yy[va]

            p_tr_x = CatPool(Xtr, yx_tr)
            p_va_x = CatPool(Xva, yx_va)
            p_tr_y = CatPool(Xtr, yy_tr)
            p_va_y = CatPool(Xva, yy_va)

            params = dict(
                iterations=self.config.ITERATIONS, learning_rate=self.config.LR, 
                depth=self.config.DEPTH, l2_leaf_reg=self.config.L2,
                random_seed=self.config.SEED, task_type=task_type, devices=devices,
                loss_function="RMSE", early_stopping_rounds=self.config.EARLY, verbose=200
            )

            model_x = CatBoostRegressor(**params)
            model_x.fit(p_tr_x, eval_set=p_va_x, verbose=200)

            model_y = CatBoostRegressor(**params)
            model_y.fit(p_tr_y, eval_set=p_va_y, verbose=200)

            # Validation predictions
            pred_rx = model_x.predict(Xva)  # residual x
            pred_ry = model_y.predict(Xva)  # residual y

            if use_residual:
                if base_x is None or base_y is None:
                    raise ValueError("use_residual=True cần base_x, base_y.")
                bx_va = base_x[va]; by_va = base_y[va]
                # residual -> absolute
                px_abs = np.clip(pred_rx + bx_va, 0.0, 120.0)
                py_abs = np.clip(pred_ry + by_va, 0.0, 53.3)
                # ground-truth absolute
                yx_abs = yx_va + bx_va
                yy_abs = yy_va + by_va
            else:
                px_abs = np.clip(pred_rx, 0.0, 120.0)
                py_abs = np.clip(pred_ry, 0.0, 53.3)
                yx_abs = yx_va
                yy_abs = yy_va

            # Use official Kaggle scoring function
            # Create DataFrames for scoring function
            solution_df = pd.DataFrame({
                'id': [f"fold_{i}_row_{j}" for j in range(len(yx_abs))],
                'x': yx_abs,
                'y': yy_abs
            })
            submission_df = pd.DataFrame({
                'id': [f"fold_{i}_row_{j}" for j in range(len(px_abs))],
                'x': px_abs,
                'y': py_abs
            })
            rmse = score(solution_df, submission_df, 'id')
            print(f"Fold {i} RMSE: {rmse:.5f}")

            models_x.append(model_x)
            models_y.append(model_y)
            fold_rmse.append(rmse)

        print("\nPer-fold RMSE:", [f"{v:.5f}" for v in fold_rmse])
        print(f"Mean ± std: {np.mean(fold_rmse):.5f} ± {np.std(fold_rmse):.5f}")
        return models_x, models_y, fold_rmse


class XGBoostTrainer:
    # Handles XGBoost model training with cross-validation
    def __init__(self, config):
        self.config = config
    
    def train_xgboost_folds(self, X, yx, yy, ids_group=None,
                           base_x=None, base_y=None, use_residual=False):
        folds = []
        if self.config.USE_GROUP_KFOLD and ids_group is not None:
            print("Using GroupKFold by (game_id,play_id,nfl_id).")
            gkf = GroupKFold(n_splits=self.config.N_FOLDS)
            for tr, va in gkf.split(X, groups=ids_group):
                folds.append((tr, va))
        else:
            print("Using plain KFold.")
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
            folds = list(kf.split(X))

        models_x, models_y, fold_rmse = [], [], []

        for i, (tr, va) in enumerate(folds, 1):
            print(f"\nXGBoost Fold {i}/{self.config.N_FOLDS} — train {len(tr):,} | val {len(va):,}")

            Xtr, Xva = X[tr], X[va]
            yx_tr, yx_va = yx[tr], yx[va]
            yy_tr, yy_va = yy[tr], yy[va]

            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.config.SEED,
                'n_jobs': -1,
                'verbosity': 0
            }

            model_x = xgb.XGBRegressor(**params, n_estimators=getattr(self.config, 'XGB_N_ESTIMATORS', 1000))
            model_x.fit(Xtr, yx_tr, eval_set=[(Xva, yx_va)], early_stopping_rounds=100, verbose=False)

            model_y = xgb.XGBRegressor(**params, n_estimators=getattr(self.config, 'XGB_N_ESTIMATORS', 1000))
            model_y.fit(Xtr, yy_tr, eval_set=[(Xva, yy_va)], early_stopping_rounds=100, verbose=False)

            # Validation predictions
            pred_rx = model_x.predict(Xva)
            pred_ry = model_y.predict(Xva)

            if use_residual:
                if base_x is None or base_y is None:
                    raise ValueError("use_residual=True requires base_x, base_y.")
                bx_va = base_x[va]; by_va = base_y[va]
                px_abs = np.clip(pred_rx + bx_va, 0.0, 120.0)
                py_abs = np.clip(pred_ry + by_va, 0.0, 53.3)
                yx_abs = yx_va + bx_va
                yy_abs = yy_va + by_va
            else:
                px_abs = np.clip(pred_rx, 0.0, 120.0)
                py_abs = np.clip(pred_ry, 0.0, 53.3)
                yx_abs = yx_va
                yy_abs = yy_va

            # Create DataFrames for scoring function
            solution_df = pd.DataFrame({
                'id': [f"xgb_fold_{i}_row_{j}" for j in range(len(yx_abs))],
                'x': yx_abs,
                'y': yy_abs
            })
            submission_df = pd.DataFrame({
                'id': [f"xgb_fold_{i}_row_{j}" for j in range(len(px_abs))],
                'x': px_abs,
                'y': py_abs
            })
            rmse = score(solution_df, submission_df, 'id')
            print(f"XGBoost Fold {i} RMSE: {rmse:.5f}")

            models_x.append(model_x)
            models_y.append(model_y)
            fold_rmse.append(rmse)

        print("\nXGBoost Per-fold RMSE:", [f"{v:.5f}" for v in fold_rmse])
        print(f"XGBoost Mean ± std: {np.mean(fold_rmse):.5f} ± {np.std(fold_rmse):.5f}")
        return models_x, models_y, fold_rmse


class LightGBMTrainer:
    # Handles LightGBM model training with cross-validation
    def __init__(self, config):
        self.config = config
    
    def train_lightgbm_folds(self, X, yx, yy, ids_group=None,
                            base_x=None, base_y=None, use_residual=False):
        folds = []
        if self.config.USE_GROUP_KFOLD and ids_group is not None:
            print("Using GroupKFold by (game_id,play_id,nfl_id).")
            gkf = GroupKFold(n_splits=self.config.N_FOLDS)
            for tr, va in gkf.split(X, groups=ids_group):
                folds.append((tr, va))
        else:
            print("Using plain KFold.")
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
            folds = list(kf.split(X))

        models_x, models_y, fold_rmse = [], [], []

        for i, (tr, va) in enumerate(folds, 1):
            print(f"\nLightGBM Fold {i}/{self.config.N_FOLDS} — train {len(tr):,} | val {len(va):,}")

            Xtr, Xva = X[tr], X[va]
            yx_tr, yx_va = yx[tr], yx[va]
            yy_tr, yy_va = yy[tr], yy[va]

            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'random_state': self.config.SEED,
                'n_jobs': -1,
                'verbosity': -1
            }

            model_x = lgb.LGBMRegressor(
                **params, 
                n_estimators=getattr(self.config, 'LGB_N_ESTIMATORS', 1000)
            )
            model_x.fit(
                Xtr, yx_tr, 
                eval_set=[(Xva, yx_va)], 
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            model_y = lgb.LGBMRegressor(
                **params, 
                n_estimators=getattr(self.config, 'LGB_N_ESTIMATORS', 1000)
            )
            model_y.fit(
                Xtr, yy_tr, 
                eval_set=[(Xva, yy_va)], 
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            # Validation predictions
            pred_rx = model_x.predict(Xva)
            pred_ry = model_y.predict(Xva)

            if use_residual:
                if base_x is None or base_y is None:
                    raise ValueError("use_residual=True requires base_x, base_y.")
                bx_va = base_x[va]; by_va = base_y[va]
                px_abs = np.clip(pred_rx + bx_va, 0.0, 120.0)
                py_abs = np.clip(pred_ry + by_va, 0.0, 53.3)
                yx_abs = yx_va + bx_va
                yy_abs = yy_va + by_va
            else:
                px_abs = np.clip(pred_rx, 0.0, 120.0)
                py_abs = np.clip(pred_ry, 0.0, 53.3)
                yx_abs = yx_va
                yy_abs = yy_va

            # Create DataFrames for scoring function
            solution_df = pd.DataFrame({
                'id': [f"lgb_fold_{i}_row_{j}" for j in range(len(yx_abs))],
                'x': yx_abs,
                'y': yy_abs
            })
            submission_df = pd.DataFrame({
                'id': [f"lgb_fold_{i}_row_{j}" for j in range(len(px_abs))],
                'x': px_abs,
                'y': py_abs
            })
            rmse = score(solution_df, submission_df, 'id')
            print(f"LightGBM Fold {i} RMSE: {rmse:.5f}")

            models_x.append(model_x)
            models_y.append(model_y)
            fold_rmse.append(rmse)

        print("\nLightGBM Per-fold RMSE:", [f"{v:.5f}" for v in fold_rmse])
        print(f"LightGBM Mean ± std: {np.mean(fold_rmse):.5f} ± {np.std(fold_rmse):.5f}")
        return models_x, models_y, fold_rmse


class EnsembleTrainer:
    # Handles ensemble training with residual learning
    def __init__(self, config):
        self.config = config
        self.catboost_trainer = CatBoostTrainer(config)
        self.xgboost_trainer = XGBoostTrainer(config)
        self.lightgbm_trainer = LightGBMTrainer(config)
        self.neural_network_trainer = NeuralNetworkTrainer(config)
    
    def train_residual_ensemble(self, X, yx, yy, ids_group=None, base_x=None, base_y=None):
        print("\n=== Training Residual Learning Ensemble ===")
        
        # Step 1: Train CatBoost as the primary model
        print("Training primary CatBoost model...")
        cat_models_x, cat_models_y, cat_rmse = self.catboost_trainer.train_catboost_folds(
            X, yx, yy, ids_group, base_x=base_x, base_y=base_y, use_residual=True
        )
        
        # Step 2: Get CatBoost out-of-fold predictions
        cat_oof_x = np.zeros(len(X))
        cat_oof_y = np.zeros(len(X))
        
        folds = []
        if self.config.USE_GROUP_KFOLD and ids_group is not None:
            gkf = GroupKFold(n_splits=self.config.N_FOLDS)
            for tr, va in gkf.split(X, groups=ids_group):
                folds.append((tr, va))
        else:
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
            folds = list(kf.split(X))
        
        for i, (tr, va) in enumerate(folds):
            Xva = X[va]
            cat_oof_x[va] = cat_models_x[i].predict(Xva)
            cat_oof_y[va] = cat_models_y[i].predict(Xva)
        
        # Step 3: Calculate residuals (errors) from CatBoost predictions
        residual_x = yx - cat_oof_x
        residual_y = yy - cat_oof_y
        
        print(f"CatBoost RMSE: {np.mean(cat_rmse):.5f}")
        print(f"Residual X std: {np.std(residual_x):.5f}, Residual Y std: {np.std(residual_y):.5f}")
        
        # Step 4: Train XGBoost and LightGBM to predict residuals
        print("Training XGBoost on residuals...")
        xgb_models_x, xgb_models_y, xgb_rmse = self.xgboost_trainer.train_xgboost_folds(
            X, residual_x, residual_y, ids_group
        )
        
        print("Training LightGBM on residuals...")
        lgb_models_x, lgb_models_y, lgb_rmse = self.lightgbm_trainer.train_lightgbm_folds(
            X, residual_x, residual_y, ids_group
        )
        
        # Step 5: Get residual predictions from XGBoost and LightGBM
        xgb_residual_x = np.zeros(len(X))
        xgb_residual_y = np.zeros(len(X))
        lgb_residual_x = np.zeros(len(X))
        lgb_residual_y = np.zeros(len(X))
        
        for i, (tr, va) in enumerate(folds):
            Xva = X[va]
            xgb_residual_x[va] = xgb_models_x[i].predict(Xva)
            xgb_residual_y[va] = xgb_models_y[i].predict(Xva)
            lgb_residual_x[va] = lgb_models_x[i].predict(Xva)
            lgb_residual_y[va] = lgb_models_y[i].predict(Xva)
        
        # Step 6: Combine predictions (CatBoost + weighted residuals)
        # Weight residuals based on CV RMSE (inverse-RMSE weighting)
        eps = 1e-6
        xgb_w = 1.0 / (np.mean(xgb_rmse) + eps)
        lgb_w = 1.0 / (np.mean(lgb_rmse) + eps)
        wsum = xgb_w + lgb_w
        xgb_w /= wsum; lgb_w /= wsum
        combined_residual_x = xgb_w * xgb_residual_x + lgb_w * lgb_residual_x
        combined_residual_y = xgb_w * xgb_residual_y + lgb_w * lgb_residual_y
        
        # Final ensemble prediction = CatBoost + residual corrections
        ensemble_pred_x = cat_oof_x + combined_residual_x
        ensemble_pred_y = cat_oof_y + combined_residual_y
        
        # Convert back to absolute positions if we have baseline
        if base_x is not None and base_y is not None:
            # Add baseline back to get absolute positions
            ensemble_abs_x = ensemble_pred_x + base_x
            ensemble_abs_y = ensemble_pred_y + base_y
            target_abs_x = yx + base_x
            target_abs_y = yy + base_y
        else:
            ensemble_abs_x = ensemble_pred_x
            ensemble_abs_y = ensemble_pred_y
            target_abs_x = yx
            target_abs_y = yy
        
        # Apply field constraints
        ensemble_abs_x = np.clip(ensemble_abs_x, 0.0, 120.0)
        ensemble_abs_y = np.clip(ensemble_abs_y, 0.0, 53.3)
        
        # Calculate ensemble RMSE using absolute positions
        ensemble_rmse = np.sqrt(0.5 * (
            mean_squared_error(target_abs_x, ensemble_abs_x) + 
            mean_squared_error(target_abs_y, ensemble_abs_y)
        ))
        
        print(f"\nResidual Learning Performance:")
        print(f"CatBoost (Primary) RMSE: {np.mean(cat_rmse):.5f}")
        print(f"XGBoost (Residual) MSE: {np.mean([mean_squared_error(residual_x, xgb_residual_x), mean_squared_error(residual_y, xgb_residual_y)]):.5f}")
        print(f"LightGBM (Residual) MSE: {np.mean([mean_squared_error(residual_x, lgb_residual_x), mean_squared_error(residual_y, lgb_residual_y)]):.5f}")
        print(f"Residual Ensemble RMSE: {ensemble_rmse:.5f}")
        
        return {
            'primary_model': {
                'catboost': (cat_models_x, cat_models_y)
            },
            'residual_models': {
                'xgboost': (xgb_models_x, xgb_models_y),
                'lightgbm': (lgb_models_x, lgb_models_y)
            },
            'residual_weights': {
                'xgboost': float(xgb_w),
                'lightgbm': float(lgb_w)
            },
            'rmse_scores': {
                'catboost': np.mean(cat_rmse),
                'ensemble': ensemble_rmse
            }
        }
    
    def train_stacking_ensemble(self, X, yx, yy, ids_group=None):
        """Train stacking ensemble with neural network integration"""
        print("\n=== Training Stacking Ensemble with Neural Network ===")
        
        # Create folds
        folds = []
        if self.config.USE_GROUP_KFOLD and ids_group is not None:
            gkf = GroupKFold(n_splits=self.config.N_FOLDS)
            for tr, va in gkf.split(X, groups=ids_group):
                folds.append((tr, va))
        else:
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED)
            folds = list(kf.split(X))
        
        # Initialize out-of-fold predictions
        oof_catboost_x = np.zeros(len(X))
        oof_catboost_y = np.zeros(len(X))
        oof_xgboost_x = np.zeros(len(X))
        oof_xgboost_y = np.zeros(len(X))
        oof_lightgbm_x = np.zeros(len(X))
        oof_lightgbm_y = np.zeros(len(X))
        oof_neural_net_x = np.zeros(len(X))
        oof_neural_net_y = np.zeros(len(X))
        
        # Store models
        catboost_models_x, catboost_models_y = [], []
        xgboost_models_x, xgboost_models_y = [], []
        lightgbm_models_x, lightgbm_models_y = [], []
        neural_net_models = []
        
        # Track individual model performance
        catboost_rmse = []
        xgboost_rmse = []
        lightgbm_rmse = []
        neural_net_rmse = []
        
        # Feature columns for neural network
        feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\nTraining Fold {fold_idx + 1}/{self.config.N_FOLDS}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            yx_train, yx_val = yx[train_idx], yx[val_idx]
            yy_train, yy_val = yy[train_idx], yy[val_idx]
            
            # Train CatBoost
            print("Training CatBoost...")
            cat_result_x = self.catboost_trainer.train_catboost_folds(
                X_train, yx_train, yy_train, None
            )
            cat_result_y = self.catboost_trainer.train_catboost_folds(
                X_train, yy_train, yx_train, None
            )
            
            catboost_models_x.extend(cat_result_x['models'])
            catboost_models_y.extend(cat_result_y['models'])
            
            # Predict on validation set
            cat_pred_x = np.mean([model.predict(X_val) for model in cat_result_x['models']], axis=0)
            cat_pred_y = np.mean([model.predict(X_val) for model in cat_result_y['models']], axis=0)
            
            oof_catboost_x[val_idx] = cat_pred_x
            oof_catboost_y[val_idx] = cat_pred_y
            
            cat_rmse = np.sqrt(0.5 * (mean_squared_error(yx_val, cat_pred_x) + mean_squared_error(yy_val, cat_pred_y)))
            catboost_rmse.append(cat_rmse)
            
            # Train XGBoost
            print("Training XGBoost...")
            xgb_result_x = self.xgboost_trainer.train_xgboost_folds(
                X_train, yx_train, yy_train, None
            )
            xgb_result_y = self.xgboost_trainer.train_xgboost_folds(
                X_train, yy_train, yx_train, None
            )
            
            xgboost_models_x.extend(xgb_result_x['models'])
            xgboost_models_y.extend(xgb_result_y['models'])
            
            # Predict on validation set
            xgb_pred_x = np.mean([model.predict(xgb.DMatrix(X_val)) for model in xgb_result_x['models']], axis=0)
            xgb_pred_y = np.mean([model.predict(xgb.DMatrix(X_val)) for model in xgb_result_y['models']], axis=0)
            
            oof_xgboost_x[val_idx] = xgb_pred_x
            oof_xgboost_y[val_idx] = xgb_pred_y
            
            xgb_rmse = np.sqrt(0.5 * (mean_squared_error(yx_val, xgb_pred_x) + mean_squared_error(yy_val, xgb_pred_y)))
            xgboost_rmse.append(xgb_rmse)
            
            # Train LightGBM
            print("Training LightGBM...")
            lgb_result_x = self.lightgbm_trainer.train_lightgbm_folds(
                X_train, yx_train, yy_train, None
            )
            lgb_result_y = self.lightgbm_trainer.train_lightgbm_folds(
                X_train, yy_train, yx_train, None
            )
            
            lightgbm_models_x.extend(lgb_result_x['models'])
            lightgbm_models_y.extend(lgb_result_y['models'])
            
            # Predict on validation set
            lgb_pred_x = np.mean([model.predict(X_val) for model in lgb_result_x['models']], axis=0)
            lgb_pred_y = np.mean([model.predict(X_val) for model in lgb_result_y['models']], axis=0)
            
            oof_lightgbm_x[val_idx] = lgb_pred_x
            oof_lightgbm_y[val_idx] = lgb_pred_y
            
            lgb_rmse = np.sqrt(0.5 * (mean_squared_error(yx_val, lgb_pred_x) + mean_squared_error(yy_val, lgb_pred_y)))
            lightgbm_rmse.append(lgb_rmse)
            
            # Train Neural Network
            print("Training Neural Network...")
            nn_result = self.neural_network_trainer.train_neural_network_folds(
                X_train, yx_train, yy_train, None, feature_cols
            )
            
            neural_net_models.extend(nn_result['models'])
            
            # Predict on validation set with neural network
            if nn_result['models']:
                nn_preds = []
                for model in nn_result['models']:
                    model.eval()
                    with torch.no_grad():
                        # Create dummy sequence data for prediction
                        val_df = pd.DataFrame(X_val, columns=feature_cols)
                        val_df['target_x'] = yx_val
                        val_df['target_y'] = yy_val
                        val_df['game_id'] = range(len(val_df))
                        val_df['play_id'] = 0
                        val_df['nfl_id'] = 0
                        val_df['frame_id'] = range(len(val_df))
                        
                        X_val_seq, _ = self.neural_network_trainer.prepare_sequence_data(
                            val_df, feature_cols, self.config.NN_WINDOW_SIZE
                        )
                        
                        if len(X_val_seq) > 0:
                            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.config.DEVICE)
                            pred = model(X_val_tensor).cpu().numpy()
                            nn_preds.append(pred)
                
                if nn_preds:
                    nn_pred = np.mean(nn_preds, axis=0)
                    # Map sequence predictions back to validation indices
                    if len(nn_pred) == len(val_idx):
                        oof_neural_net_x[val_idx] = nn_pred[:, 0]
                        oof_neural_net_y[val_idx] = nn_pred[:, 1]
                    else:
                        # Fallback: use mean prediction for all validation samples
                        oof_neural_net_x[val_idx] = np.mean(nn_pred[:, 0])
                        oof_neural_net_y[val_idx] = np.mean(nn_pred[:, 1])
                    
                    nn_rmse = np.sqrt(0.5 * (
                        mean_squared_error(yx_val, oof_neural_net_x[val_idx]) + 
                        mean_squared_error(yy_val, oof_neural_net_y[val_idx])
                    ))
                else:
                    nn_rmse = float('inf')
            else:
                nn_rmse = float('inf')
            
            neural_net_rmse.append(nn_rmse)
            
            print(f"Fold {fold_idx + 1} RMSE - CatBoost: {cat_rmse:.4f}, XGBoost: {xgb_rmse:.4f}, "
                  f"LightGBM: {lgb_rmse:.4f}, Neural Net: {nn_rmse:.4f}")
        
        # Create meta-features
        meta_features = np.column_stack([
            oof_catboost_x, oof_catboost_y,
            oof_xgboost_x, oof_xgboost_y,
            oof_lightgbm_x, oof_lightgbm_y,
            oof_neural_net_x, oof_neural_net_y
        ])
        
        # Train meta-learners
        meta_learner_x = Ridge(alpha=1.0)
        meta_learner_y = Ridge(alpha=1.0)
        
        meta_learner_x.fit(meta_features, yx)
        meta_learner_y.fit(meta_features, yy)
        
        # Final ensemble predictions
        ensemble_pred_x = meta_learner_x.predict(meta_features)
        ensemble_pred_y = meta_learner_y.predict(meta_features)
        
        ensemble_rmse = np.sqrt(0.5 * (
            mean_squared_error(yx, ensemble_pred_x) + 
            mean_squared_error(yy, ensemble_pred_y)
        ))
        
        print(f"\nStacking Ensemble Performance:")
        print(f"CatBoost RMSE: {np.mean(catboost_rmse):.5f}")
        print(f"XGBoost RMSE: {np.mean(xgboost_rmse):.5f}")
        print(f"LightGBM RMSE: {np.mean(lightgbm_rmse):.5f}")
        print(f"Neural Net RMSE: {np.mean(neural_net_rmse):.5f}")
        print(f"Ensemble RMSE: {ensemble_rmse:.5f}")
        
        return {
            'base_models': {
                'catboost': (catboost_models_x, catboost_models_y),
                'xgboost': (xgboost_models_x, xgboost_models_y),
                'lightgbm': (lightgbm_models_x, lightgbm_models_y),
                'neural_net': neural_net_models
            },
            'meta_learners': {
                'x': meta_learner_x,
                'y': meta_learner_y
            },
            'rmse_scores': {
                'catboost': np.mean(catboost_rmse),
                'xgboost': np.mean(xgboost_rmse),
                'lightgbm': np.mean(lightgbm_rmse),
                'neural_net': np.mean(neural_net_rmse),
                'ensemble': ensemble_rmse
            }
        }

class NFLPredictor:
    def __init__(self):
        self.config = NFLConfig()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer()
        self.gnn_processor = GNNLiteProcessor(self.config)
        self.training_builder = TrainingDataBuilder()
        self.trainer = CatBoostTrainer(self.config)
        self.ensemble_trainer = EnsembleTrainer(self.config)
        
        # Ensure save directory exists
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
    
    def prepare_training_data(self):
        # Load data
        tr_in, tr_out = self.data_loader.load_all_train()
        
        # Feature engineering
        print("\nEngineering features on train…")
        tr_in = self.feature_engineer.engineer_advanced_features(tr_in)
        tr_in = self.feature_engineer.add_sequence_features(tr_in)
        tr_in = self.feature_engineer.add_ensemble_features(tr_in)
        
        # GNN-lite processing
        print("Computing neighbor embeddings (train)…")
        gnn_tr = self.gnn_processor.compute_neighbor_embeddings(tr_in)
        
        # Build training rows
        train_df = self.training_builder.create_training_rows(tr_in, tr_out)
        print("Train rows (pre-merge GNN):", train_df.shape)
        
        # Merge GNN features
        train_df = train_df.merge(gnn_tr, on=["game_id","play_id","nfl_id"], how="left")
        
        # Physics baseline and residuals
        bx, by = self.training_builder.physics_baseline(
            train_df["x"].values, train_df["y"].values,
            train_df["velocity_x"].values, train_df["velocity_y"].values,
            train_df["delta_t"].values
        )
        
        # Use official Kaggle scoring function for physics baseline
        solution_df = pd.DataFrame({
            'id': [f"baseline_row_{i}" for i in range(len(train_df))],
            'x': train_df["target_x"].values,
            'y': train_df["target_y"].values
        })
        submission_df = pd.DataFrame({
            'id': [f"baseline_row_{i}" for i in range(len(train_df))],
            'x': bx,
            'y': by
        })
        base_rmse = score(solution_df, submission_df, 'id')
        print(f"Physics baseline RMSE: {base_rmse:.5f}")
        
        train_df["base_x"] = bx
        train_df["base_y"] = by
        train_df["res_x"]  = train_df["target_x"] - train_df["base_x"]
        train_df["res_y"]  = train_df["target_y"] - train_df["base_y"]
        
        return train_df, base_rmse
    
    def prepare_test_data(self, feat_cols):
        # Load test data
        te_in, te_tpl = self.data_loader.load_test_data()
        
        # Feature engineering
        te_in = self.feature_engineer.engineer_advanced_features(te_in)
        te_in = self.feature_engineer.add_sequence_features(te_in)
        te_in = self.feature_engineer.add_ensemble_features(te_in)
        
        # GNN-lite processing
        print("Computing neighbor embeddings (test)…")
        gnn_te = self.gnn_processor.compute_neighbor_embeddings(te_in)
        
        # Aggregate test data
        agg_te = (
            te_in.sort_values(["game_id","play_id","nfl_id","frame_id"])
                 .groupby(["game_id","play_id","nfl_id"], as_index=False)
                 .tail(1)
                 .rename(columns={"frame_id":"last_frame_id"})
        )
        
        te = te_tpl.merge(agg_te, on=["game_id","play_id","nfl_id"], how="left")
        te = te.merge(gnn_te, on=["game_id","play_id","nfl_id"], how="left")
        
        te["delta_frames"] = (te["frame_id"] - te["last_frame_id"]).clip(lower=0).astype(float)
        te["delta_t"] = te["delta_frames"] / 10.0
        
        # Ensure all feature columns exist
        for c in feat_cols:
            if c not in te.columns:
                te[c] = 0.0
        te.loc[:, feat_cols] = te[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        
        return te
    
    def train_and_predict(self):
        # Prepare training data
        train_df, base_rmse = self.prepare_training_data()
        
        # Build feature list
        feat_cols = self.training_builder.build_feature_list(train_df)
        print(f"Using {len(feat_cols)} features (incl. GNN-lite).")
        
        # Clean and prepare matrices
        df_train = train_df.dropna(subset=feat_cols + ["res_x","res_y"]).reset_index(drop=True)
        df_train.loc[:, feat_cols] = (
            df_train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        )
        X  = df_train[feat_cols].values.astype(np.float32)
        yx = df_train["res_x"].values.astype(np.float32)   # residual
        yy = df_train["res_y"].values.astype(np.float32)
        base_vec_x = df_train["base_x"].values.astype(np.float32)
        base_vec_y = df_train["base_y"].values.astype(np.float32)
        
        # Groups for cross-validation
        groups = (df_train["game_id"].astype(str) + "_" +
                  df_train["play_id"].astype(str) + "_" +
                  df_train["nfl_id"].astype(str)).values if self.config.USE_GROUP_KFOLD else None
        
        # Train residual ensemble instead of just CatBoost
        print("\nTraining residual ensemble...")
        ensemble_results = self.ensemble_trainer.train_residual_ensemble(
            X, yx, yy, ids_group=groups, base_x=base_vec_x, base_y=base_vec_y
        )
        
        # Extract models from ensemble results
        models_x = ensemble_results['primary_model']['catboost'][0]
        models_y = ensemble_results['primary_model']['catboost'][1]
        residual_models = ensemble_results.get('residual_models', {})
        residual_weights = ensemble_results.get('residual_weights', {})
        fold_rmse = [ensemble_results['rmse_scores']['ensemble']] * self.config.N_FOLDS
        
        # Save models
        with open(self.config.SAVE_DIR/"catboost_models_5fold_gnnlite.pkl", "wb") as f:
            pickle.dump(
                {"models_x": models_x, "models_y": models_y, "features": feat_cols, "cv_rmse": fold_rmse},
                f
            )
        print("Saved:", self.config.SAVE_DIR/"catboost_models_5fold_gnnlite.pkl")
        
        # Prepare test data
        te = self.prepare_test_data(feat_cols)
        Xtest = te[feat_cols].values.astype(np.float32)
        
        # Baseline test predictions
        tbx, tby = self.training_builder.physics_baseline(
            te["x"].values, te["y"].values,
            te["velocity_x"].values, te["velocity_y"].values,
            te["delta_t"].values
        )
        
        # Ensemble predictions
        # CatBoost residual predictions
        cat_residual_rx = np.mean([m.predict(Xtest) for m in models_x], axis=0)
        cat_residual_ry = np.mean([m.predict(Xtest) for m in models_y], axis=0)

        # Residual corrections from XGBoost and LightGBM (if available)
        if residual_models:
            xgb_models_x, xgb_models_y = residual_models.get('xgboost', ([], []))
            lgb_models_x, lgb_models_y = residual_models.get('lightgbm', ([], []))
            xgb_residual_rx = np.mean([m.predict(Xtest) for m in xgb_models_x], axis=0) if xgb_models_x else 0.0
            xgb_residual_ry = np.mean([m.predict(Xtest) for m in xgb_models_y], axis=0) if xgb_models_y else 0.0
            lgb_residual_rx = np.mean([m.predict(Xtest) for m in lgb_models_x], axis=0) if lgb_models_x else 0.0
            lgb_residual_ry = np.mean([m.predict(Xtest) for m in lgb_models_y], axis=0) if lgb_models_y else 0.0

            # Weighted combination using learned residual weights
            w_xgb = float(residual_weights.get('xgboost', 0.5))
            w_lgb = float(residual_weights.get('lightgbm', 0.5))
            ws = max(w_xgb + w_lgb, 1e-6)
            w_xgb /= ws; w_lgb /= ws
            combined_residual_rx = w_xgb * xgb_residual_rx + w_lgb * lgb_residual_rx
            combined_residual_ry = w_xgb * xgb_residual_ry + w_lgb * lgb_residual_ry
        else:
            combined_residual_rx = 0.0
            combined_residual_ry = 0.0

        # Final residual + baseline -> absolute coordinates
        ensemble_residual_rx = cat_residual_rx + combined_residual_rx
        ensemble_residual_ry = cat_residual_ry + combined_residual_ry
        pred_x  = np.clip(ensemble_residual_rx + tbx, 0.0, 120.0)
        pred_y  = np.clip(ensemble_residual_ry + tby, 0.0, 53.3)
        
        # Create submission
        sub = pd.DataFrame({
            "id": (te["game_id"].astype(str) + "_" +
                   te["play_id"].astype(str) + "_" +
                   te["nfl_id"].astype(str) + "_" +
                   te["frame_id"].astype(str)),
            "x": pred_x,
            "y": pred_y
        })
        sub.to_csv(self.config.SAVE_DIR/"submission.csv", index=False)

        # Save residual ensemble package for reproducibility
        with open(self.config.SAVE_DIR/"residual_ensemble_models_5fold_gnnlite.pkl", "wb") as f:
            pickle.dump(
                {
                    "primary_catboost_x": models_x,
                    "primary_catboost_y": models_y,
                    "residual_xgboost_x": residual_models.get('xgboost', ([], []))[0] if residual_models else [],
                    "residual_xgboost_y": residual_models.get('xgboost', ([], []))[1] if residual_models else [],
                    "residual_lightgbm_x": residual_models.get('lightgbm', ([], []))[0] if residual_models else [],
                    "residual_lightgbm_y": residual_models.get('lightgbm', ([], []))[1] if residual_models else [],
                    "residual_weights": residual_weights,
                    "features": feat_cols,
                    "cv_rmse": fold_rmse
                },
                f
            )
        print("Saved:", self.config.SAVE_DIR/"residual_ensemble_models_5fold_gnnlite.pkl")
        print("Saved submission:", self.config.SAVE_DIR/"submission.csv")
        
        # Final results
        print("\nCV Mean ± std (absolute, with baseline added back):", f"{np.mean(fold_rmse):.5f} ± {np.std(fold_rmse):.5f}")
        print("Physics baseline (absolute):", f"{base_rmse:.5f}")
        
        return sub, fold_rmse, base_rmse

def main():
    predictor = NFLPredictor()
    submission, cv_scores, baseline_rmse = predictor.train_and_predict()
    
    print("\n" + "="*50)
    print("NFL Big Data Bowl 2026 - Training Complete!")
    print(f"Final CV Score: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    print(f"Baseline Improvement: {baseline_rmse - np.mean(cv_scores):.5f}")
    print("="*50)

if __name__ == "__main__":
    main()