import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
import pickle
import re

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from multiprocessing import Pool as MultiprocessingPool, cpu_count

warnings.filterwarnings('ignore')

# CONFIG

class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Where to persist trained artifacts (models, scalers, route objects)
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    # Toggle saving/loading of artifacts
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = True
    LOAD_DIR = '/kaggle/input/geometric-gnn/pytorch/default/1/outputs/models'
    # Behavior control:
    # - If pretrained artifacts are present at LOAD_DIR and SKIP_TRAIN_IF_PRETRAINED is True,
    #   we will load them and skip training (submission-friendly, avoids timeouts).
    # - Set environment variable FORCE_TRAIN=1 to override and run training regardless.
    SKIP_TRAIN_IF_PRETRAINED = True
    FORCE_TRAIN = os.environ.get("FORCE_TRAIN", "0") == "1"
    # If local test files exist (./data/test.csv and ./data/test_input.csv),
    # automatically generate submission.parquet for notebook convenience after
    # training or after loading pretrained artifacts.
    AUTO_GENERATE_SUBMISSION_LOCAL = True
    # Optional outputs
    WRITE_SUBMISSION_PARQUET = True
    SAVE_PRETRAIN_META = True
  
    SEED = 42
    N_FOLDS = 10
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3
    
    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94
    
    # Transformer hyperparameters
    N_HEADS = 4   # Transformer attention head
    N_LAYERS = 2  # Transformer encoder layer
    
    # ResidualMLP head hyperparameters
    MLP_HIDDEN_DIM = 256 # MLP head hidden dimension
    N_RES_BLOCKS = 2     # Number of residual blocks
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    K_NEIGH = 5
    RADIUS = 20.0
    TAU = 6.0
    N_ROUTE_CLUSTERS = 7
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = False
    if DEBUG:
        N_FOLDS = 2

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# GEOMETRIC BASELINE

def compute_geometric_endpoint(df):
    """
    Compute where each player SHOULD end up based on geometry.
    This is the deterministic part - no learning needed.
    """
    df = df.copy()
    
    # Time to play end
    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0
    else:
        t_total = 3.0
    
    df['time_to_endpoint'] = t_total
    
    # Initialize with momentum (default rule)
    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total
    
    # Rule 1: Targeted Receivers converge to ball
    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']
        
        # Rule 2: Defenders mirror receivers (maintain offset)
        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror
        
        df.loc[coverage_mask, 'geo_endpoint_x'] = (
            df.loc[coverage_mask, 'ball_land_x'] + 
            df.loc[coverage_mask, 'mirror_offset_x'].fillna(0)
        )
        df.loc[coverage_mask, 'geo_endpoint_y'] = (
            df.loc[coverage_mask, 'ball_land_y'] + 
            df.loc[coverage_mask, 'mirror_offset_y'].fillna(0)
        )
    
    # Clip to field
    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    
    return df

def add_geometric_features(df):
    """Add features that describe the geometric solution"""
    df = compute_geometric_endpoint(df)
    
    # Vector to geometric endpoint
    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)
    
    # Required velocity to reach geometric endpoint
    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t
    
    # Current velocity vs required
    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )
    
    # Required constant acceleration (a = 2*Δx/t²)
    t_sq = t * t
    df['geo_required_ax'] = 2 * df['geo_vector_x'] / t_sq
    df['geo_required_ay'] = 2 * df['geo_vector_y'] / t_sq
    df['geo_required_ax'] = df['geo_required_ax'].clip(-10, 10)
    df['geo_required_ay'] = df['geo_required_ay'].clip(-10, 10)
    
    # Alignment with geometric path
    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)
    
    # Role-specific geometric quality
    df['geo_receiver_urgency'] = df['is_receiver'] * df['geo_distance'] / (t + 0.1)
    df['geo_defender_coupling'] = df['is_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50) + 1.0))
    
    return df

# Feature engineering

def get_velocity(speed, direction_deg):
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

def get_opponent_features(input_df):
    features = []

    for (gid, pid), group in tqdm(
        input_df.groupby(['game_id', 'play_id']), desc="Opponents", leave=False
    ):
        last = group.sort_values('frame_id').groupby('nfl_id').last()
        
        if len(last) < 2:
            continue
            
        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values
        
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])
        
        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side
            
            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
                'mirror_wr_dist': 50.0,
            }
            
            if not opp_mask.any():
                features.append(feat)
                continue
            
            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))
            
            if len(distances) == 0:
                features.append(feat)
                continue
                
            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()
            
            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])
            
            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])
            
            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))
                
                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]
                    
                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])
                    
                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]
            
            features.append(feat)
    
    return pd.DataFrame(features)

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=True):
    route_features = []
    
    for (gid, pid, nid), group in tqdm(
        input_df.groupby(['game_id', 'play_id', 'nfl_id']), desc="Routes", leave=False
    ):
        traj = group.sort_values('frame_id').tail(5)
        
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + 
                              (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })
    
    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=Config.N_ROUTE_CLUSTERS, random_state=Config.SEED, n_init=10)
        route_df['route_pattern'] = kmeans.fit_predict(X_scaled)
        return route_df, kmeans, scaler
    else:
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)
        return route_df

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH, 
                                radius=Config.RADIUS, tau=Config.TAU):
    """Neighborhood-based interaction embeddings (GNN-like)."""
    print("Computing neighborhood interaction embeddings...")
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", 
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))
    
    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb", 
            "velocity_x": "vx_nb", "velocity_y": "vy_nb", 
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )
    
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]
    
    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)
    
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]
    
    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])
    
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()
    
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")
    
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", 
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    
    return ag

def add_advanced_features(input_df, is_training=True, route_kmeans=None, route_scaler=None):
    """Create advanced features: opponent stats, route patterns, and neighborhood embeddings."""
    print("Step 2: Advanced features...")

    # Opponent features
    opp_features = get_opponent_features(input_df)
    input_df = input_df.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    # Route patterns
    if is_training:
        route_features, route_kmeans, route_scaler = extract_route_patterns(input_df)
    else:
        route_features = extract_route_patterns(input_df, route_kmeans, route_scaler, fit=False)
    
    if not route_features.empty:
        input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    # GNN features
    gnn_features = compute_neighbor_embeddings(input_df)
    input_df = input_df.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')

    # Pressure and mirror features
    if 'nearest_opp_dist' in input_df.columns:
        input_df['pressure'] = 1 / np.maximum(input_df['nearest_opp_dist'], 0.5)
        input_df['under_pressure'] = (input_df['nearest_opp_dist'] < 3).astype(int)
        input_df['pressure_x_speed'] = input_df['pressure'] * input_df['s']

    if 'mirror_wr_vx' in input_df.columns:
        s_safe = np.maximum(input_df['s'], 0.1)
        input_df['mirror_similarity'] = (
            input_df['velocity_x'] * input_df['mirror_wr_vx'] +
            input_df['velocity_y'] * input_df['mirror_wr_vy']
        ) / s_safe
        input_df['mirror_offset_dist'] = np.sqrt(
            input_df['mirror_offset_x']**2 + input_df['mirror_offset_y']**2
        )
        input_df['mirror_alignment'] = input_df['mirror_similarity'] * input_df['role_defensive_coverage']

    if is_training:
        return input_df, route_kmeans, route_scaler
    return input_df

# Sequence preparation with geometric features

def prepare_sequences_geometric(input_df, output_df=None, test_template=None, 
                                is_training=True, window_size=10,
                                route_kmeans=None, route_scaler=None):
    """Prepare sequences with geometric and engineered features (167 features)."""

    print("Preparing geometric sequences")
    
    input_df = input_df.copy()
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    print("Step 1: Base features...")
    
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    height_parts = input_df['player_height'].str.split('-', expand=True)
    input_df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    input_df['bmi'] = (input_df['player_weight'] / (input_df['height_inches']**2)) * 703
    
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.cos(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.sin(dir_rad)
    
    input_df['speed_squared'] = input_df['s'] ** 2
    input_df['accel_magnitude'] = np.sqrt(input_df['acceleration_x']**2 + input_df['acceleration_y']**2)
    input_df['momentum_x'] = input_df['velocity_x'] * input_df['player_weight']
    input_df['momentum_y'] = input_df['velocity_y'] * input_df['player_weight']
    input_df['kinetic_energy'] = 0.5 * input_df['player_weight'] * input_df['speed_squared']
    
    input_df['orientation_diff'] = np.abs(input_df['o'] - input_df['dir'])
    input_df['orientation_diff'] = np.minimum(input_df['orientation_diff'], 360 - input_df['orientation_diff'])
    
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['role_targeted_receiver'] = input_df['is_receiver']
    input_df['role_defensive_coverage'] = input_df['is_coverage']
    input_df['role_passer'] = input_df['is_passer']
    input_df['side_offense'] = input_df['is_offense']
    
    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['dist_to_ball'] = input_df['distance_to_ball']
        input_df['dist_squared'] = input_df['distance_to_ball'] ** 2
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed_ball'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
        input_df['velocity_toward_ball'] = (
            input_df['velocity_x'] * np.cos(input_df['angle_to_ball']) + 
            input_df['velocity_y'] * np.sin(input_df['angle_to_ball'])
        )
        input_df['velocity_alignment'] = np.cos(input_df['angle_to_ball'] - dir_rad)
        input_df['angle_diff'] = np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball']))
        input_df['angle_diff'] = np.minimum(input_df['angle_diff'], 360 - input_df['angle_diff'])
    
    if is_training:
        input_df, route_kmeans, route_scaler = add_advanced_features(input_df, is_training=True)
    else:
        input_df = add_advanced_features(input_df, is_training=False, route_kmeans=route_kmeans, route_scaler=route_scaler)
    
    print("Step 3: Temporal features...")
    
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in input_df.columns:
                input_df[f'{col}_lag{lag}'] = input_df.groupby(gcols)[col].shift(lag)
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in input_df.columns:
                input_df[f'{col}_rolling_mean_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                input_df[f'{col}_rolling_std_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )
    
    for col in ['velocity_x', 'velocity_y']:
        if col in input_df.columns:
            input_df[f'{col}_delta'] = input_df.groupby(gcols)[col].diff()
    
    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    
    print("Step 4: Time features...")
    
    if 'num_frames_output' in input_df.columns:
        max_frames = input_df['num_frames_output']
        
        input_df['max_play_duration'] = max_frames / 10.0
        input_df['frame_time'] = input_df['frame_id'] / 10.0
        input_df['progress_ratio'] = input_df['frame_id'] / np.maximum(max_frames, 1)
        input_df['time_remaining'] = (max_frames - input_df['frame_id']) / 10.0
        input_df['frames_remaining'] = max_frames - input_df['frame_id']
        
        input_df['expected_x_at_ball'] = input_df['x'] + input_df['velocity_x'] * input_df['frame_time']
        input_df['expected_y_at_ball'] = input_df['y'] + input_df['velocity_y'] * input_df['frame_time']
        
        if 'ball_land_x' in input_df.columns:
            input_df['error_from_ball_x'] = input_df['expected_x_at_ball'] - input_df['ball_land_x']
            input_df['error_from_ball_y'] = input_df['expected_y_at_ball'] - input_df['ball_land_y']
            input_df['error_from_ball'] = np.sqrt(
                input_df['error_from_ball_x']**2 + input_df['error_from_ball_y']**2
            )
            
            input_df['weighted_dist_by_time'] = input_df['dist_to_ball'] / (input_df['frame_time'] + 0.1)
            input_df['dist_scaled_by_progress'] = input_df['dist_to_ball'] * (1 - input_df['progress_ratio'])
        
        input_df['time_squared'] = input_df['frame_time'] ** 2
        input_df['velocity_x_progress'] = input_df['velocity_x'] * input_df['progress_ratio']
        input_df['velocity_y_progress'] = input_df['velocity_y'] * input_df['progress_ratio']
        input_df['speed_scaled_by_time_left'] = input_df['s'] * input_df['time_remaining']
        
        input_df['actual_play_length'] = max_frames
        input_df['length_ratio'] = max_frames / 30.0
    
    # Add geometric features
    print("Step 5: Geometric endpoint features...")
    input_df = add_geometric_features(input_df)
    
    print("Step 6: Building feature list...")
    
    # Base feature set
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y',
        'player_height_feet', 'player_weight', 'height_inches', 'bmi',
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'speed_squared', 'accel_magnitude', 'orientation_diff',
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
        'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball', 
        'ball_direction_x', 'ball_direction_y', 'closing_speed_ball',
        'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
        'pressure', 'under_pressure', 'pressure_x_speed', 
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.append(f'{col}_rolling_mean_{window}')
            feature_cols.append(f'{col}_rolling_std_{window}')
    
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    feature_cols.extend(['velocity_x_ema', 'velocity_y_ema', 'speed_ema'])
    
    feature_cols.extend([
        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball', 
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time', 
        'velocity_x_progress', 'velocity_y_progress', 'dist_scaled_by_progress',
        'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
    ])
    
    # Add 13 geometric features
    feature_cols.extend([
        'geo_endpoint_x', 'geo_endpoint_y',
        'geo_vector_x', 'geo_vector_y', 'geo_distance',
        'geo_required_vx', 'geo_required_vy',
        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error',
        'geo_required_ax', 'geo_required_ay',
        'geo_alignment',
    ])
    
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    print(f"Using {len(feature_cols)} features")
    
    print("Step 7: Creating sequences...")
    
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    target_rows = output_df if is_training else test_template
    target_groups = target_rows[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = [], [], [], [], []
    geo_endpoints_x, geo_endpoints_y = [], []
    
    for _, row in tqdm(target_groups.iterrows(), total=len(target_groups), desc="Creating sequences"):
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        
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
        
        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values
        
        if np.isnan(seq).any():
            if is_training:
                continue
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        
        # Store geometric endpoint for this player
        geo_x = input_window.iloc[-1]['geo_endpoint_x']
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_x.append(geo_x)
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_y.append(geo_y)
        
        if is_training:
            out_grp = output_df[
                (output_df['game_id']==row['game_id']) &
                (output_df['play_id']==row['play_id']) &
                (output_df['nfl_id']==row['nfl_id'])
            ].sort_values('frame_id')
            
            last_x = input_window.iloc[-1]['x']
            last_y = input_window.iloc[-1]['y']
            
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_frame_ids.append(out_grp['frame_id'].values)
        
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })
    
    print(f"Created {len(sequences)} sequences")
    
    if is_training:
        return (sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, 
                geo_endpoints_x, geo_endpoints_y, route_kmeans, route_scaler)
    return sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y

# Model architecture (ST-Transformer with Residual MLP head)
class ResidualBlock(nn.Module):
    """Residual block: feed-forward network with skip connection."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Pre-normalization
        return x + self.ffn(self.norm(x))

class ResidualMLPHead(nn.Module):
    """Residual MLP prediction head."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_res_blocks=2, dropout=0.2):
        super().__init__()
        # 1) Project context_dim to mlp_hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        # 2) Residual blocks operating at hidden_dim
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_res_blocks)]
        )
        
        # 3) Final LayerNorm and output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_norm(x)
        x = self.output_layer(x)
        return x

class STTransformer(nn.Module):
    """Spatio-temporal Transformer."""
    def __init__(self, input_dim, hidden_dim, horizon, window_size, n_heads, n_layers, dropout=0.1):
        super().__init__()
        config = Config()  # MLP head hyperparameters
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # 1) Feature embedding
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 2) Learned positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim)) 
        self.embed_dropout = nn.Dropout(dropout)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4) Attention pooling
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim)) 

        # 5) Prediction head (ResidualMLPHead)
        self.head = ResidualMLPHead(
            input_dim=hidden_dim,                   # 128
            hidden_dim=config.MLP_HIDDEN_DIM,       # 256
            output_dim=horizon * 2,                 # 188
            n_res_blocks=config.N_RES_BLOCKS,       # 2
            dropout=0.2
        )

    def forward(self, x):
        B, S, _ = x.shape
        x_embed = self.input_projection(x) 
        x = x_embed + self.pos_embed[:, :S, :] 
        x = self.embed_dropout(x)
        
        h = self.transformer_encoder(x) 

        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.squeeze(1) 

        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        
        out = torch.cumsum(out, dim=1)
        
        return out


# Loss (Temporal Huber)

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
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)

# Training

def prepare_targets(batch_dx, batch_dy, max_h):
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

def compute_val_rmse(model, X_val, y_val_dx, y_val_dy, horizon, device, batch_size=256):
    """
    Compute validation RMSE (Root Mean Squared Error) for 2D trajectory prediction.
    Returns the average Euclidean distance error across all predictions.
    """
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32)).to(device)
            
            # Get predictions
            pred = model(bx).cpu().numpy()  # Shape: (batch, horizon, 2)
            
            # Process each sample in the batch
            for j, idx in enumerate(range(i, end)):
                dx_true = y_val_dx[idx]
                dy_true = y_val_dy[idx]
                n_steps = len(dx_true)
                
                # Extract predictions for this sample
                dx_pred = pred[j, :n_steps, 0]
                dy_pred = pred[j, :n_steps, 1]
                
                # Compute Euclidean distance for each time step
                sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
                all_errors.extend(sq_errors)
    
    # Return RMSE
    return np.sqrt(np.mean(all_errors))

def train_model(X_train, y_train_dx, y_train_dy, X_val, y_val_dx, y_val_dy, 
                input_dim, horizon, config):
    device = config.DEVICE
    
    model = STTransformer(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        horizon=horizon,
        window_size=config.WINDOW_SIZE,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS
    ).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            horizon
        )
        train_batches.append((bx, by, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            horizon
        )
        val_batches.append((bx, by, bm))
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pred = model(bx)
            loss = criterion(pred, by, bm)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())
        
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
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss

import kaggle_evaluation.nfl_inference_server
import warnings
import polars as pl # Import Polars for the predict function signature

# Move all utility functions (get_velocity, height_to_feet, compute_geometric_endpoint, etc.)
# and the Config/set_seed outside the class but above it, or make them static methods.

class NFLPredictor:
    def __init__(self):
        warnings.filterwarnings('ignore')
        config = Config() # Use the existing Config class
        set_seed(config.SEED)
        self.config = config

        # If configured, try to load pre-saved artifacts to skip training and data prep
        load_dir = Path(config.LOAD_DIR) if config.LOAD_DIR is not None else config.MODEL_DIR
        route_ok = (load_dir / "route_kmeans.pkl").exists() and (load_dir / "route_scaler.pkl").exists()
        # Accept multiple filename variants: model_foldN.pt, model_foldN.pth, or model_foldN (no extension)
        _candidates = []
        for pat in ["model_fold*.pt", "model_fold*.pth", "model_fold*"]:
            _candidates.extend(load_dir.glob(pat))
        # Deduplicate and keep only files
        _unique = {}
        for p in _candidates:
            if p.is_file() and p.name.startswith("model_fold"):
                _unique[str(p)] = p
        def _fold_key(path_obj):
            m = re.search(r"model_fold(\d+)", path_obj.name)
            return int(m.group(1)) if m else 0
        model_files = sorted(_unique.values(), key=_fold_key)
        # Only load pretrained artifacts and exit early if configured to skip training
        if (
            config.LOAD_ARTIFACTS
            and route_ok
            and len(model_files) > 0
            and config.SKIP_TRAIN_IF_PRETRAINED
            and not config.FORCE_TRAIN
        ):
            print(f"\nLoading pretrained artifacts from {load_dir}...")
            self.models = []
            self.scalers = []
            # Load route objects
            try:
                with open(load_dir / "route_kmeans.pkl", "rb") as f:
                    self.route_kmeans = pickle.load(f)
                with open(load_dir / "route_scaler.pkl", "rb") as f:
                    self.route_scaler = pickle.load(f)
            except Exception as e:
                print("Failed to load route artifacts:", e)

            # Infer input_dim from any available scaler, else default
            input_dim = 167
            try:
                any_scaler = next((p for p in load_dir.glob("scaler_fold*.pkl")), None)
                if any_scaler is not None:
                    with open(any_scaler, "rb") as f:
                        s1 = pickle.load(f)
                    input_dim = int(getattr(s1, "mean_", np.zeros(167)).shape[0])
            except Exception:
                pass

            # Load all available model folds
            for model_path in model_files:
                # Parse fold id
                try:
                    fold_str = model_path.name.replace("model_fold", "").replace(".pt", "")
                    fold = int(fold_str)
                except Exception:
                    fold = None

                scaler = None
                if fold is not None:
                    scaler_path = load_dir / f"scaler_fold{fold}.pkl"
                    if scaler_path.exists():
                        try:
                            with open(scaler_path, "rb") as f:
                                scaler = pickle.load(f)
                        except Exception:
                            scaler = None

                model = STTransformer(
                    input_dim=input_dim,
                    hidden_dim=config.HIDDEN_DIM,
                    horizon=config.MAX_FUTURE_HORIZON,
                    window_size=config.WINDOW_SIZE,
                    n_heads=config.N_HEADS,
                    n_layers=config.N_LAYERS
                ).to(config.DEVICE)
                try:
                    state = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state)
                except Exception as e:
                    print(f"Failed to load model from {model_path.name}:", e)

                model.to(config.DEVICE)
                model.eval()
                self.models.append(model)
                self.scalers.append(scaler)

            print(f"Loaded {len(self.models)} model(s) from {load_dir}. Ready for inference.")
            print("Pretrained artifacts found and SKIP_TRAIN_IF_PRETRAINED=True. Skipping training.")
            # Auto-generate local submission.parquet (for notebook convenience) if local test files exist
            try:
                if getattr(self.config, 'AUTO_GENERATE_SUBMISSION_LOCAL', True):
                    test_path_local = Path('./data/test.csv')
                    test_input_path_local = Path('./data/test_input.csv')
                    if test_path_local.exists() and test_input_path_local.exists():
                        print("[4/4] Generating local submission.parquet from pretrained models...")
                        import polars as pl
                        test_df = pl.read_csv(str(test_path_local))
                        test_input_df = pl.read_csv(str(test_input_path_local))
                        _ = self.predict(test_df, test_input_df)
            except Exception as e:
                print("Warning: failed to auto-generate local submission with pretrained models:", e)
            return
        elif (
            config.LOAD_ARTIFACTS
            and route_ok
            and len(model_files) > 0
            and (not config.SKIP_TRAIN_IF_PRETRAINED or config.FORCE_TRAIN)
        ):
            # Artifacts exist but we are asked to continue to training
            reason = (
                "FORCE_TRAIN=1 env var set" if config.FORCE_TRAIN else "SKIP_TRAIN_IF_PRETRAINED=False"
            )
            print(f"Pretrained artifacts detected at {load_dir}, but {reason}; proceeding to training.")

        # If not loading, proceed with training and data preparation
        # 1. Load Data
        print("[1/4] Loading data for training...")
        train_input_files = [config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
        train_output_files = [config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
        if Config.DEBUG:            
            # Load a small amount of data to quickly get play IDs
            sample_train_input = pd.read_csv(train_input_files[0])
            sample_train_output = pd.read_csv(train_output_files[0])
            # Get the first 100 unique plays (game_id, play_id combinations)
            sample_plays = sample_train_output[['game_id', 'play_id']].drop_duplicates().head(100)
            # Filter the full data for only these plays
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
            # Perform the actual filtering to get a small, but complete, set of plays
            train_input = train_input.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            train_output = train_output.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            print(f"Reduced to {len(sample_plays)} unique plays.")
        else:
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])

        # 2. Prepare Sequences and Train Feature Objects
        print("\n[2/4] Preparing geometric sequences and feature scalers...")
        result = prepare_sequences_geometric(
            train_input, train_output, is_training=True, window_size=config.WINDOW_SIZE
        )
        sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, geo_x, geo_y, route_kmeans, route_scaler = result

        # Store feature objects for later inference
        self.route_kmeans = route_kmeans
        self.route_scaler = route_scaler
        
        sequences = list(sequences)
        targets_dx = list(targets_dx)
        targets_dy = list(targets_dy)

        # Ensure model dir exists
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Train Models
        print("\n[3/4] Training geometric models...")
        groups = np.array([d['game_id'] for d in sequence_ids])
        gkf = GroupKFold(n_splits=config.N_FOLDS)

        self.models, self.scalers = [], []
        fold_losses = []
        fold_rmses = []

        for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
            # ... (rest of the fold training logic, exactly as in the original main()) ...
            X_tr = [sequences[i] for i in tr]
            X_va = [sequences[i] for i in va]
            y_tr_dx = [targets_dx[i] for i in tr]
            y_va_dx = [targets_dx[i] for i in va]
            y_tr_dy = [targets_dy[i] for i in tr]
            y_va_dy = [targets_dy[i] for i in va]
            scaler = StandardScaler()
            scaler.fit(np.vstack([s for s in X_tr]))
            X_tr_sc = [scaler.transform(s) for s in X_tr]
            X_va_sc = [scaler.transform(s) for s in X_va]
            model, loss = train_model(
                X_tr_sc, y_tr_dx, y_tr_dy,
                X_va_sc, y_va_dx, y_va_dy,
                X_tr[0].shape[-1], config.MAX_FUTURE_HORIZON, config
            )
            
            # Compute validation RMSE
            val_rmse = compute_val_rmse(
                model, X_va_sc, y_va_dx, y_va_dy, 
                config.MAX_FUTURE_HORIZON, config.DEVICE, config.BATCH_SIZE
            )
            
            self.models.append(model)
            self.scalers.append(scaler)
            fold_losses.append(loss)
            fold_rmses.append(val_rmse)
            
            print(f"\nFold {fold} - Loss: {loss:.5f}, Validation RMSE: {val_rmse:.5f}")
            # Persist fold artifacts if requested
            if config.SAVE_ARTIFACTS:
                try:
                    model_path = config.MODEL_DIR / f"model_fold{fold}.pt"
                    scaler_path = config.MODEL_DIR / f"scaler_fold{fold}.pkl"
                    # save model state dict on CPU to avoid device issues
                    state = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(state, str(model_path))
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scaler, f)
                    print(f"  Saved artifacts for fold {fold} -> {model_path.name}, {scaler_path.name}")
                except Exception as e:
                    print(f"  Warning: failed to save artifacts for fold {fold}:", e)

        # Print summary statistics across all folds
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        avg_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        avg_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        
        print(f"Average Loss:           {avg_loss:.5f} ± {std_loss:.5f}")
        print(f"Average Validation RMSE: {avg_rmse:.5f} ± {std_rmse:.5f}")
        print(f"\nPer-Fold Results:")
        for i, (loss, rmse) in enumerate(zip(fold_losses, fold_rmses), 1):
            print(f"  Fold {i}: Loss={loss:.5f}, RMSE={rmse:.5f}")
        print("="*60 + "\n")

        # Optionally save cross-validation meta for pretraining/reference
        if config.SAVE_ARTIFACTS and getattr(config, 'SAVE_PRETRAIN_META', False):
            try:
                meta = {
                    'fold_losses': fold_losses,
                    'fold_rmses': fold_rmses,
                    'avg_loss': float(avg_loss),
                    'std_loss': float(std_loss),
                    'avg_rmse': float(avg_rmse),
                    'std_rmse': float(std_rmse),
                    'n_features': int(sequences[0].shape[1]) if sequences else None,
                }
                with open(config.MODEL_DIR / "pretrain_meta.pkl", "wb") as f:
                    pickle.dump(meta, f)
                print("Saved pretrain meta -> pretrain_meta.pkl")
            except Exception as e:
                print("Warning: failed to save pretrain meta:", e)

        # Ensure all models are in eval mode (best practice)
        # Optionally persist route clustering objects
        if config.SAVE_ARTIFACTS:
            try:
                with open(config.MODEL_DIR / "route_kmeans.pkl", "wb") as f:
                    pickle.dump(self.route_kmeans, f)
                with open(config.MODEL_DIR / "route_scaler.pkl", "wb") as f:
                    pickle.dump(self.route_scaler, f)
                print(f"Saved route artifacts -> route_kmeans.pkl, route_scaler.pkl")
            except Exception as e:
                print("Warning: failed to save route artifacts:", e)

        for model in self.models:
            model.eval()
        print("\nModel ready for inference.")

        # Auto-generate local submission.parquet (for notebook convenience) if local test files exist
        try:
            if getattr(self.config, 'AUTO_GENERATE_SUBMISSION_LOCAL', True):
                test_path_local = Path('./data/test.csv')
                test_input_path_local = Path('./data/test_input.csv')
                if test_path_local.exists() and test_input_path_local.exists():
                    print("[4/4] Generating local submission.parquet after training...")
                    import polars as pl
                    test_df = pl.read_csv(str(test_path_local))
                    test_input_df = pl.read_csv(str(test_input_path_local))
                    _ = self.predict(test_df, test_input_df)
        except Exception as e:
            print("Warning: failed to auto-generate local submission after training:", e)


    def predict(self, test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
        """Inference function called by the API for each time step."""
        
        # Convert Polars DataFrames to Pandas, as the original notebook uses Pandas extensively
        test_input_pd = test_input.to_pandas()
        test_template_pd = test.to_pandas()
        
        # 1. Prepare Test Sequences
        # Use the stored feature objects (route_kmeans, route_scaler) for inference
        test_seq, test_ids, test_geo_x, test_geo_y = prepare_sequences_geometric(
            test_input_pd, test_template=test_template_pd, is_training=False,
            window_size=self.config.WINDOW_SIZE,
            route_kmeans=self.route_kmeans, route_scaler=self.route_scaler
        )

        X_test = list(test_seq)
        x_last = np.array([s[-1, 0] for s in X_test])
        y_last = np.array([s[-1, 1] for s in X_test])

        # 2. Ensemble Prediction
        all_preds = []
        H = self.config.MAX_FUTURE_HORIZON

        for model, sc in zip(self.models, self.scalers):
            X_sc = [sc.transform(s) if sc is not None else s for s in X_test]
            X_t = torch.tensor(np.stack(X_sc).astype(np.float32)).to(self.config.DEVICE)

            with torch.no_grad():
                preds = model(X_t).cpu().numpy()

            all_preds.append(preds)

        ens_preds = np.mean(all_preds, axis=0)

        # 3. Format Submission (Corrected for API)
        rows = []
        for i, sid in enumerate(test_ids):
            # The template DataFrame 'test' contains the info needed to map predictions
            fids = test_template_pd[
                (test_template_pd['game_id'] == sid['game_id']) &
                (test_template_pd['play_id'] == sid['play_id']) &
                (test_template_pd['nfl_id'] == sid['nfl_id'])
            ]['frame_id'].sort_values().tolist()
            
            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                px = np.clip(x_last[i] + ens_preds[i, tt, 0], 0, self.config.FIELD_X_MAX)
                py = np.clip(y_last[i] + ens_preds[i, tt, 1], 0, self.config.FIELD_Y_MAX)

                # DO NOT include 'id' in the rows dict or final DataFrame
                # The API will handle the row IDs based on the provided 'test' DataFrame
                rows.append({
                    'x': px,
                    'y': py
                })

        # The final returned DataFrame MUST only contain 'x' and 'y' columns, 
        # matching the order of rows in the input 'test' DataFrame.
        submission = pd.DataFrame(rows)

        # Optionally write submission file to disk for convenience
        if getattr(self.config, 'WRITE_SUBMISSION_PARQUET', False):
            try:
                out_path = self.config.OUTPUT_DIR / "submission.parquet"
                root_path = Path("submission.parquet")
                submission.to_parquet(out_path, index=False)
                # Also write to project root to satisfy notebooks that expect ./submission.parquet
                try:
                    submission.to_parquet(root_path, index=False)
                except Exception:
                    pass
                print(f"Saved submission parquet -> {out_path}")
                if root_path.exists():
                    print(f"Saved submission parquet (duplicate) -> {root_path}")
            except Exception as e:
                # Fallback to CSV if parquet engine is unavailable
                try:
                    out_csv = self.config.OUTPUT_DIR / "submission.csv"
                    root_csv = Path("submission.csv")
                    submission.to_csv(out_csv, index=False)
                    try:
                        submission.to_csv(root_csv, index=False)
                    except Exception:
                        pass
                    print(f"Saved submission CSV -> {out_csv} (parquet unavailable: {e})")
                    if root_csv.exists():
                        print(f"Saved submission CSV (duplicate) -> {root_csv}")
                except Exception as e2:
                    print(f"Warning: failed to save submission file: {e2}")
        return submission

# This must be outside the class and is what the API calls
predictor = NFLPredictor()

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """The function the API calls."""
    return predictor.predict(test, test_input)

# Set up the Kaggle evaluation server interface as required by the competition.
# When your notebook is run on the hidden test set, inference_server.serve must be called within 10 minutes
# of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your
# model you can do so during the very first predict call, which does not have the usual 5 minute response deadline.
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # In competition rerun environment, start serving immediately.
    inference_server.serve()
else:
    # For local testing, point the gateway to the public competition files directory.
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
