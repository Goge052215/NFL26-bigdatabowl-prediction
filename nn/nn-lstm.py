import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
import warnings
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

# Configs
class Config:
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("working")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    SEED = 42
    N_FOLDS = 5

    BATCH_SIZE = 256
    EPOCHS = 1000
    PATIENCE = 100
    LEARNING_RATE = 5e-4
    
    WINDOW_SIZE = 8
    HIDDEN_DIM = 256
    MAX_FUTURE_HORIZON = 120
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    # GNN-lite parameters
    K_NEIGH = 5
    RADIUS = 15.0
    TAU = 6.0
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=Config.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# feature engineering
def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

# GNN-lite neighbor embedding computations
class GNNLiteProcessor:
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
        if Config.RADIUS is not None:
            tmp = tmp[tmp["dist"] <= Config.RADIUS]

        # ally / opp flag
        tmp["is_ally"] = (tmp["player_side_nb"].fillna("") == tmp["player_side"].fillna("")).astype(np.float32)

        # rank by distance (keep top-K)
        keys = ["game_id","play_id","nfl_id"]
        tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
        if Config.K_NEIGH is not None:
            tmp = tmp[tmp["rnk"] <= float(Config.K_NEIGH)]

        # attention weights: softmax(-dist/tau) within group
        tmp["w"] = np.exp(-tmp["dist"] / float(Config.TAU))
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
            ag[c] = ag[c].fillna(Config.RADIUS if Config.RADIUS is not None else 30.0)

        return ag

def add_advanced_features(df):
    df = df.copy()
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    # GROUP 1: Distance Rate Features (3)
    if 'distance_to_ball' in df.columns:
        df['distance_to_ball_change'] = df.groupby(gcols)['distance_to_ball'].diff().fillna(0)
        df['distance_to_ball_accel'] = df.groupby(gcols)['distance_to_ball_change'].diff().fillna(0)
        df['time_to_intercept'] = (
            df['distance_to_ball'] / 
            (np.abs(df['distance_to_ball_change']) + 0.1)).clip(0, 10)
    
    # GROUP 2: Target Alignment Features (3)
    if 'ball_direction_x' in df.columns:
        df['velocity_alignment'] = (
            df['velocity_x'] * df['ball_direction_x'] +
            df['velocity_y'] * df['ball_direction_y']
        )
        df['velocity_perpendicular'] = (
            df['velocity_x'] * (-df['ball_direction_y']) +
            df['velocity_y'] * df['ball_direction_x']
        )
        if 'acceleration_x' in df.columns:
            df['accel_alignment'] = (
                df['acceleration_x'] * df['ball_direction_x'] +
                df['acceleration_y'] * df['ball_direction_y']
            )
    
    # GROUP 3: Multi-Window Rolling (24)
    for window in [3, 5, 10]:
        for col in ['velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_roll{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_std{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                ).fillna(0)
    
    # GROUP 4: Extended Lag Features (8)
    for lag in [4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(gcols)[col].shift(lag).fillna(0)
    
    # GROUP 5: Velocity Change Features (4)
    if 'velocity_x' in df.columns:
        df['velocity_x_change'] = df.groupby(gcols)['velocity_x'].diff().fillna(0)
        df['velocity_y_change'] = df.groupby(gcols)['velocity_y'].diff().fillna(0)
        df['speed_change'] = df.groupby(gcols)['s'].diff().fillna(0)
        df['direction_change'] = df.groupby(gcols)['dir'].diff().fillna(0)
        df['direction_change'] = df['direction_change'].apply(
            lambda x: x if abs(x) < 180 else x - 360 * np.sign(x)
        )
    
    # GROUP 6: Field Position Features (4)
    df['dist_from_left'] = df['y']
    df['dist_from_right'] = 53.3 - df['y']
    df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
    df['dist_from_endzone'] = np.minimum(df['x'], 120 - df['x'])
    
    # GROUP 7: Role-Specific Features (3)
    if 'is_receiver' in df.columns and 'velocity_alignment' in df.columns:
        df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
        df['receiver_deviation'] = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0))
    if 'is_coverage' in df.columns and 'closing_speed' in df.columns:
        df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']
    
    # GROUP 8: Time Features (2)
    df['frames_elapsed'] = df.groupby(gcols).cumcount()
    df['normalized_time'] = df.groupby(gcols)['frames_elapsed'].transform(
        lambda x: x / (x.max() + 1)
    )
    
    print(f"Total features after enhancement: {len(df.columns)}")
    
    return df

def prepare_sequences_with_advanced_features(input_df, output_df=None, test_template=None, 
                                            is_training=True, window_size=Config.WINDOW_SIZE):
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES")
    print(f"Window size: {window_size}")
    
    input_df = input_df.copy()
    
    # BASIC FEATURES
    print("Step 1/3: Adding basic features...")
    
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    delta_t = 0.1
    input_df['velocity_x'] = (input_df['s'] + 0.5 * input_df['a'] * delta_t) * np.sin(dir_rad)
    input_df['velocity_y'] = (input_df['s'] + 0.5 * input_df['a'] * delta_t) * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.sin(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.cos(dir_rad)
    
    # Roles
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    
    # Physics
    mass_kg = input_df['player_weight'].fillna(200.0) / 2.20462
    input_df['momentum_x'] = input_df['velocity_x'] * mass_kg
    input_df['momentum_y'] = input_df['velocity_y'] * mass_kg
    input_df['kinetic_energy'] = 0.5 * mass_kg * (input_df['s'] ** 2)
    
    # Ball features
    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
    
    # Sort for temporal
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    # Original lag features (1-3)
    for lag in [1, 2, 3]:
        input_df[f'x_lag{lag}'] = input_df.groupby(gcols)['x'].shift(lag)
        input_df[f'y_lag{lag}'] = input_df.groupby(gcols)['y'].shift(lag)
        input_df[f'velocity_x_lag{lag}'] = input_df.groupby(gcols)['velocity_x'].shift(lag)
        input_df[f'velocity_y_lag{lag}'] = input_df.groupby(gcols)['velocity_y'].shift(lag)
    
    # EMA features
    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    
    # ADVANCED FEATURES
    print("Step 2/3: Adding advanced features...")
    input_df = add_advanced_features(input_df)
    
    # GNN LITE FEATURES
    print("Step 2.5/3: Adding GNN Lite features...")
    gnn_processor = GNNLiteProcessor()
    gnn_features = gnn_processor.compute_neighbor_embeddings(input_df)
    
    # Merge GNN features back to input_df
    input_df = input_df.merge(
        gnn_features,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    # Fill NaN values for GNN features
    gnn_cols = [c for c in input_df.columns if c.startswith('gnn_')]
    for col in gnn_cols:
        if col in ['gnn_ally_cnt', 'gnn_opp_cnt']:
            input_df[col] = input_df[col].fillna(0.0)
        elif 'mean' in col or 'dx' in col or 'dy' in col or 'dvx' in col or 'dvy' in col:
            input_df[col] = input_df[col].fillna(0.0)
        else:  # distance features
            input_df[col] = input_df[col].fillna(Config.RADIUS)
    
    # FEATURE LIST
    print("Step 3/3: Creating sequences...")
    
    feature_cols = [
        # Core (9)
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y',
        
        # Player (2)
        'player_height_feet', 'player_weight',
        
        # Motion (7)
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        
        # Roles (5)
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        
        # Ball (5)
        'distance_to_ball', 'angle_to_ball', 'ball_direction_x', 'ball_direction_y', 'closing_speed',
        
        # Original temporal (15)
        'x_lag1', 'y_lag1', 'velocity_x_lag1', 'velocity_y_lag1',
        'x_lag2', 'y_lag2', 'velocity_x_lag2', 'velocity_y_lag2',
        'x_lag3', 'y_lag3', 'velocity_x_lag3', 'velocity_y_lag3',
        'velocity_x_ema', 'velocity_y_ema', 'speed_ema',
        
        # NEW: Distance rate (3)
        'distance_to_ball_change', 'distance_to_ball_accel', 'time_to_intercept',
        
        # NEW: Target alignment (3)
        'velocity_alignment', 'velocity_perpendicular', 'accel_alignment',
        
        # NEW: Multi-window rolling (24)
        'velocity_x_roll3', 'velocity_x_std3', 'velocity_y_roll3', 'velocity_y_std3',
        's_roll3', 's_std3', 'a_roll3', 'a_std3',
        'velocity_x_roll5', 'velocity_x_std5', 'velocity_y_roll5', 'velocity_y_std5',
        's_roll5', 's_std5', 'a_roll5', 'a_std5',
        'velocity_x_roll10', 'velocity_x_std10', 'velocity_y_roll10', 'velocity_y_std10',
        's_roll10', 's_std10', 'a_roll10', 'a_std10',
        
        # NEW: Extended lags (8)
        'x_lag4', 'y_lag4', 'velocity_x_lag4', 'velocity_y_lag4',
        'x_lag5', 'y_lag5', 'velocity_x_lag5', 'velocity_y_lag5',
        
        # NEW: Velocity changes (4)
        'velocity_x_change', 'velocity_y_change', 'speed_change', 'direction_change',
        
        # NEW: Field position (4)
        'dist_from_sideline', 'dist_from_endzone',
        
        # NEW: Role-specific (3)
        'receiver_optimality', 'receiver_deviation', 'defender_closing_speed',
        
        # NEW: Time (2)
        'frames_elapsed', 'normalized_time',
        
        # GNN LITE FEATURES (20)
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_dist_1', 'gnn_ally_dist_2', 'gnn_ally_dist_3',
        'gnn_opp_dist_1', 'gnn_opp_dist_2', 'gnn_opp_dist_3',
        'gnn_nearest_ally_dist', 'gnn_nearest_opp_dist',
        'gnn_ally_attention_sum', 'gnn_opp_attention_sum',
    ]
    
    # Filter to existing
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    print(f"Using {len(feature_cols)} features (was ~50, now ~90)")
    
    # CREATE SEQUENCES
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    target_rows = output_df if is_training else test_template
    target_groups = target_rows[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    # Pre-compute group means for faster fillna operations
    print("Pre-computing group statistics...")
    group_means = grouped.mean(numeric_only=True)
    
    # Pre-create output lookup dictionary for training
    output_lookup = {}
    if is_training:
        print("Creating output lookup dictionary...")
        for _, row in output_df.iterrows():
            key = (row['game_id'], row['play_id'], row['nfl_id'])
            if key not in output_lookup:
                output_lookup[key] = []
            output_lookup[key].append({
                'x': row['x'], 'y': row['y'], 'frame_id': row['frame_id']
            })
        
        # Sort each group by frame_id
        for key in output_lookup:
            output_lookup[key] = sorted(output_lookup[key], key=lambda x: x['frame_id'])
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = [], [], [], [], []
    
    # Convert target_groups to list of tuples for faster iteration
    target_keys = [(row['game_id'], row['play_id'], row['nfl_id']) 
                   for _, row in target_groups.iterrows()]
    
    # Pre-allocate arrays for better memory efficiency
    num_sequences = len(target_keys)
    print(f"Processing {num_sequences} sequences...")
    
    # Pre-allocate lists with estimated capacity
    sequences = []
    if is_training:
        targets_dx = []
        targets_dy = []
        targets_frame_ids = []
    sequence_ids = []
    for key in tqdm(target_keys, desc="Creating sequences"):
        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue
        
        input_window = group_df.tail(window_size)
        
        if len(input_window) < window_size:
            if is_training:
                continue
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)
        
        # Use pre-computed means for faster fillna
        if key in group_means.index:
            input_window = input_window.fillna(group_means.loc[key])
        else:
            input_window = input_window.fillna(0.0)
        
        seq = input_window[feature_cols].values
        
        if np.isnan(seq).any():
            if is_training:
                continue
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        
        if is_training and key in output_lookup:
            out_data = output_lookup[key]
            
            last_x = input_window.iloc[-1]['x']
            last_y = input_window.iloc[-1]['y']
            
            dx = np.array([d['x'] for d in out_data]) - last_x
            dy = np.array([d['y'] for d in out_data]) - last_y
            frame_ids = np.array([d['frame_id'] for d in out_data])
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_frame_ids.append(frame_ids)
        
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })
    
    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")
    
    if is_training:
        return sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids
    return sequences, sequence_ids

# loss
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L)
            huber, mask = huber * weight, mask * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)

# Conv-layered NN
class SeqModel(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        # Enhanced CNN with residual connection
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.cnn_proj = nn.Linear(64, input_dim)  # For residual connection
        
        # Bidirectional LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_dim, 192,  # Increased from 128 to 192
            num_layers=3,    # Increased from 2 to 3 layers
            batch_first=True, 
            dropout=0.2,     # Increased dropout
            bidirectional=True  # Bidirectional for better context
        )
        # Note: bidirectional=True doubles the output size to 384
        
        # Layer normalization for better gradient flow
        self.lstm_ln = nn.LayerNorm(384)  # 192 * 2 for bidirectional
        
        self.gpu_proj = nn.Linear(256, 128)
        
        # Enhanced attention pooling with more capacity
        self.attention_pooling = nn.Sequential(
            nn.Linear(384, 128),  # Updated for bidirectional LSTM output
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced multi-head attention
        self.pool_ln = nn.LayerNorm(384)  # Updated for bidirectional output
        self.pool_attn = nn.MultiheadAttention(
            384, num_heads=8,  # Increased from 4 to 8 heads
            batch_first=True,
            dropout=0.1
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, 384))  # Updated size
        
        # Enhanced prediction head with skip connections
        self.head = nn.Sequential(
            nn.Linear(384, 256),  # Updated input size
            nn.LayerNorm(256),
            nn.GELU(), 
            nn.Dropout(0.2), 
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, horizon)
        )
        
        # Learnable combination of pooling methods
        self.pooling_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        B, seq_len, features = x.shape
        
        # Enhanced CNN preprocessing with residual connection
        x_cnn = x.transpose(1, 2)  # (B, features, seq_len) for Conv1d
        cnn_out = self.cnn(x_cnn)  # (B, 64, seq_len)
        cnn_out = cnn_out.transpose(1, 2)  # (B, seq_len, 64)
        cnn_proj = self.cnn_proj(cnn_out)  # (B, seq_len, features)
        
        # Residual connection: combine original input with CNN features
        x_enhanced = x + 0.1 * cnn_proj  # Small residual weight
        
        # Bidirectional LSTM encoding
        h, (hidden_state, cell_state) = self.lstm(x_enhanced)  # Shape: (B, seq_len, 384)
        h = self.lstm_ln(h)  # Layer normalization
        
        # Method 1: Enhanced attention pooling
        attention_weights = self.attention_pooling(h)  # Shape: (B, seq_len, 1)
        context_vector_1 = torch.sum(attention_weights * h, dim=1)  # Shape: (B, 384)
        
        # Method 2: Enhanced multi-head attention pooling
        q = self.pool_query.expand(B, -1, -1)
        ctx_2, attn_weights = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        context_vector_2 = ctx_2.squeeze(1)  # Shape: (B, 384)
        
        # Learnable combination with temperature scaling
        alpha = torch.sigmoid(self.pooling_weight * 2.0)  # Temperature scaling
        context_vector = alpha * context_vector_1 + (1 - alpha) * context_vector_2
        
        # Enhanced prediction with residual connection
        out = self.head(context_vector)
        
        # Physics-informed cumulative sum with momentum
        return torch.cumsum(out, dim=1)

# Custom Dataset class for proper DataLoader usage
class NFLDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, horizon):
        self.sequences = sequences
        self.targets = targets
        self.horizon = horizon
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx].astype(np.float32))
        target = self.targets[idx]
        
        # Prepare target with padding and mask
        L = len(target)
        padded_target = np.pad(target, (0, self.horizon - L), constant_values=0).astype(np.float32)
        mask = np.zeros(self.horizon, dtype=np.float32)
        mask[:L] = 1.0
        
        return sequence, torch.tensor(padded_target), torch.tensor(mask)

# Custom collate function for variable length sequences
def collate_fn(batch):
    sequences, targets, masks = zip(*batch)
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    masks = torch.stack(masks)
    return sequences, targets, masks

def train_model(X_train, y_train, X_val, y_val, input_dim, horizon, Config):
    device = Config.DEVICE
    model = SeqModel(input_dim, horizon).to(device)
    
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Create datasets and dataloaders
    train_dataset = NFLDataset(X_train, y_train, horizon)
    val_dataset = NFLDataset(X_val, y_val, horizon)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for MPS compatibility
        pin_memory=False  # Disable for MPS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for MPS compatibility
        pin_memory=False  # Disable for MPS
    )
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_losses = []
        
        for batch_idx, (sequences, targets, masks) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            pred = model(sequences)
            loss = criterion(pred, targets, masks)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sequences, targets, masks in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                pred = model(sequences)
                val_losses.append(criterion(pred, targets, masks).item())
        
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
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

# main pipeline
def main():
    # Load
    print("\n[1/4] Loading data...")
    train_input_files = [Config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
    train_output_files = [Config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
    train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
    train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
    test_input = pd.read_csv(Config.DATA_DIR / "test_input.csv")
    test_template = pd.read_csv(Config.DATA_DIR / "test.csv")
    
    # Prepare with advanced features
    print("\n[2/4] Preparing with ADVANCED features...")
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = prepare_sequences_with_advanced_features(
        train_input, train_output, is_training=True, window_size=Config.WINDOW_SIZE
    )
    
    sequences = np.array(sequences, dtype=object)
    targets_dx = np.array(targets_dx, dtype=object)
    targets_dy = np.array(targets_dy, dtype=object)
    
    # Train
    print("\n[3/4] Training with enhanced features...")
    groups = np.array([d['game_id'] for d in sequence_ids])
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    
    models_x, models_y, scalers = [], [], []
    
    for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{Config.N_FOLDS}")
        print(f"{'='*60}")
        
        X_tr, X_va = sequences[tr], sequences[va]
        
        scaler = StandardScaler()
        scaler.fit(np.vstack([s for s in X_tr]))
        
        X_tr_sc = np.stack([scaler.transform(s) for s in X_tr])
        X_va_sc = np.stack([scaler.transform(s) for s in X_va])
        
        # Train X
        print("Training X-axis model...")
        mx, loss_x = train_model(
            X_tr_sc, targets_dx[tr], X_va_sc, targets_dx[va],
            X_tr[0].shape[-1], Config.MAX_FUTURE_HORIZON, Config
        )
        
        # Train Y
        print("Training Y-axis model...")
        my, loss_y = train_model(
            X_tr_sc, targets_dy[tr], X_va_sc, targets_dy[va],
            X_tr[0].shape[-1], Config.MAX_FUTURE_HORIZON, Config
        )
        
        models_x.append(mx)
        models_y.append(my)
        scalers.append(scaler)
        
        print(f"\nFold {fold} - X loss: {loss_x:.5f}, Y loss: {loss_y:.5f}")
    
    # Test predictions
    print("\n[4/4] Creating test predictions...")
    test_sequences, test_ids = prepare_sequences_with_advanced_features(
        test_input, test_template=test_template, is_training=False, window_size=Config.WINDOW_SIZE
    )
    
    X_test = np.array(test_sequences, dtype=object)
    x_last = np.array([s[-1, 0] for s in X_test])
    y_last = np.array([s[-1, 1] for s in X_test])
    
    # Ensemble predictions across folds
    all_dx, all_dy = [], []
    for mx, my, sc in zip(models_x, models_y, scalers):
        X_sc = np.stack([sc.transform(s) for s in X_test])
        X_t = torch.tensor(X_sc.astype(np.float32)).to(Config.DEVICE)
        
        mx.eval()
        my.eval()
        
        with torch.no_grad():
            all_dx.append(mx(X_t).cpu().numpy())
            all_dy.append(my(X_t).cpu().numpy())
    
    ens_dx = np.mean(all_dx, axis=0)
    ens_dy = np.mean(all_dy, axis=0)
    
    # Create submission
    rows = []
    H = ens_dx.shape[1]
    
    for i, sid in enumerate(test_ids):
        fids = test_template[
            (test_template['game_id'] == sid['game_id']) &
            (test_template['play_id'] == sid['play_id']) &
            (test_template['nfl_id'] == sid['nfl_id'])
        ]['frame_id'].sort_values().tolist()
        
        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            px = np.clip(x_last[i] + ens_dx[i, tt], 0, 120)
            py = np.clip(y_last[i] + ens_dy[i, tt], 0, 53.3)
            
            rows.append({
                'id': f"{sid['game_id']}_{sid['play_id']}_{sid['nfl_id']}_{fid}",
                'x': px,
                'y': py
            })
    
    submission = pd.DataFrame(rows)
    submission.to_csv("submission.csv", index=False)
    
    return submission

if __name__ == "__main__":
    main()