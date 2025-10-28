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
    # Where to persist trained artifacts (models, scalers, route objects)
    MODEL_DIR = Path("nn/models")
    MODEL_DIR.mkdir(exist_ok=True)
    # Toggle saving/loading of artifacts
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = True
    LOAD_DIR = '/Users/goge/nfl26/nn/models'
    
    SEED = 42
    N_FOLDS = 5

    BATCH_SIZE = 256
    EPOCHS = 1000
    PATIENCE = 100
    LEARNING_RATE = 4e-4
    
    WINDOW_SIZE = 8
    HIDDEN_DIM = 256
    MAX_FUTURE_HORIZON = 124  # Test this first
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    K_NEIGH = 5
    RADIUS = 20.0
    TAU = 6.0
    N_ROUTE_CLUSTERS = 7
    
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

def set_seed(seed=Config.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# ============================================================================
# GEOMETRIC BASELINE - THE BREAKTHROUGH
# ============================================================================

def compute_geometric_endpoint(df):
    """
    GEOMETRIC ENDPOINT COMPUTATION: Physics-based trajectory prediction
    ===================================================================
    
    This function computes where each player should end up based on game context and physics.
    It's the core innovation that provides deterministic baseline predictions.
    
    ALGORITHM:
    1. Initialize with momentum-based projection (default behavior for all players)
    2. Apply game-context rules:
       - Targeted receivers → converge to ball landing position
       - Defensive coverage → mirror receivers with maintained spatial offset
       - All others → continue with momentum-based projection
    3. Apply field boundary constraints (0-120 yards x, 0-53.3 yards y)
    
    PHYSICS MODEL:
    - Uses current velocity to project natural movement over time horizon
    - Time horizon based on remaining frames in play (or default 3.0 seconds)
    - Applies realistic field boundary clipping
    
    INPUT: DataFrame with player tracking data including ball landing coordinates
    OUTPUT: DataFrame with added geo_endpoint_x, geo_endpoint_y, and time_to_endpoint columns
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
    """
    GEOMETRIC FEATURE ENGINEERING: Physics-based trajectory analysis
    ================================================================
    
    This function generates ~12 geometric features that capture how well players
    are executing optimal trajectories toward their predicted endpoints.
    
    FEATURE CATEGORIES:
    1. Spatial vectors (geo_vector_x/y, geo_distance)
    2. Required velocities (geo_required_vx/vy)
    3. Velocity errors (geo_velocity_error_x/y, geo_velocity_error)
    4. Required accelerations (geo_required_ax/ay)
    5. Trajectory alignment (geo_alignment)
    6. Role-specific metrics (geo_receiver_urgency, geo_defender_coupling)
    
    PHYSICS PRINCIPLES:
    - Uses kinematic equations: v = Δx/t, a = 2Δx/t²
    - Computes alignment using dot products of velocity vectors
    - Measures role-specific performance (receiver urgency, defender coupling)
    
    INPUT: DataFrame with player tracking data
    OUTPUT: Enhanced DataFrame with geometric baseline features
    """
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
    """
    COMPREHENSIVE FEATURE ENGINEERING PIPELINE
    ==========================================
    
    This function transforms raw NFL tracking data into a rich feature set for trajectory prediction.
    The pipeline is organized into 12 feature groups, each targeting specific aspects of player movement:
    
    PROCEDURE:
    1. Data preparation: Sort by temporal order and define grouping columns
    2. Apply geometric baseline features (deterministic trajectory predictions)
    3. Extract temporal dynamics (velocity changes, acceleration patterns)
    4. Compute spatial relationships (field position, player interactions)
    5. Generate rolling statistics for temporal smoothing
    6. Create lag features for historical context
    
    INPUT: DataFrame with raw tracking data (x, y, velocity_x, velocity_y, etc.)
    OUTPUT: Enhanced DataFrame with 80+ engineered features
    """
    df = df.copy()
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']  # Grouping columns for temporal operations
    
    # ========================================================================
    # GROUP 0: GEOMETRIC BASELINE FEATURES (BREAKTHROUGH INNOVATION)
    # ========================================================================
    """
    GEOMETRIC PROCESSING: Physics-based trajectory prediction
    
    This is the key innovation that provides deterministic baseline predictions:
    - Computes where each player SHOULD end up based on game context
    - Targeted receivers → converge to ball landing position
    - Defenders → mirror receiver movements with spatial offset
    - Other players → continue with current momentum
    
    Features generated: ~15 geometric features including:
    - geo_endpoint_x/y: Predicted final positions
    - geo_required_vx/vy: Velocities needed to reach endpoints
    - geo_velocity_error: Deviation from required velocity
    - geo_alignment: How well current movement aligns with geometric path
    """
    print("  → Applying geometric baseline features...")
    df = add_geometric_features(df)
    
    # ========================================================================
    # GROUP 1: DISTANCE RATE FEATURES (3 features)
    # ========================================================================
    """
    BALL INTERACTION DYNAMICS: How players relate to ball position
    
    Captures the temporal evolution of player-ball relationships:
    - distance_to_ball_change: Rate of approach/retreat from ball
    - distance_to_ball_accel: Acceleration of approach (2nd derivative)
    - time_to_intercept: Estimated time to reach ball at current rate
    """
    if 'distance_to_ball' in df.columns:
        df['distance_to_ball_change'] = df.groupby(gcols)['distance_to_ball'].diff().fillna(0)
        df['distance_to_ball_accel'] = df.groupby(gcols)['distance_to_ball_change'].diff().fillna(0)
        df['time_to_intercept'] = (
            df['distance_to_ball'] / 
            (np.abs(df['distance_to_ball_change']) + 0.1)).clip(0, 10)
    
    # ========================================================================
    # GROUP 2: TARGET ALIGNMENT FEATURES (3 features)
    # ========================================================================
    """
    DIRECTIONAL COHERENCE: How well player movement aligns with ball trajectory
    
    Decomposes velocity into components relative to ball direction:
    - velocity_alignment: Dot product of velocity with ball direction (forward/backward)
    - velocity_perpendicular: Cross product component (lateral movement)
    - accel_alignment: Acceleration alignment with ball direction
    """
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
    
    # ========================================================================
    # GROUP 3: MULTI-WINDOW ROLLING STATISTICS (24 features)
    # ========================================================================
    """
    TEMPORAL SMOOTHING: Rolling averages and variability measures
    
    Captures movement patterns at different time scales:
    - Windows: 3, 5, 10 frames (0.3s, 0.5s, 1.0s at 10 FPS)
    - Variables: velocity_x, velocity_y, speed (s), acceleration (a)
    - Statistics: mean (trend) and std (variability)
    
    Purpose: Smooth noisy tracking data and capture multi-scale temporal patterns
    """
    for window in [3, 5, 10]:
        for col in ['velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_roll{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_std{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                ).fillna(0)
    
    # ========================================================================
    # GROUP 4: EXTENDED LAG FEATURES (8 features)
    # ========================================================================
    """
    HISTORICAL CONTEXT: Medium-term historical positions and velocities
    
    Provides context from 0.4-0.5 seconds ago:
    - Lags: 4, 5 frames (0.4s, 0.5s at 10 FPS)
    - Variables: x, y, velocity_x, velocity_y
    
    Purpose: Capture longer-term movement trends and route patterns
    """
    for lag in [4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(gcols)[col].shift(lag).fillna(0)
    
    # ========================================================================
    # GROUP 5: VELOCITY CHANGE FEATURES (4 features)
    # ========================================================================
    """
    ACCELERATION PATTERNS: First derivatives of kinematic variables
    
    Captures instantaneous changes in movement:
    - velocity_x/y_change: Linear acceleration components
    - speed_change: Change in overall speed
    - direction_change: Angular acceleration (with 360° wrap-around handling)
    """
    if 'velocity_x' in df.columns:
        df['velocity_x_change'] = df.groupby(gcols)['velocity_x'].diff().fillna(0)
        df['velocity_y_change'] = df.groupby(gcols)['velocity_y'].diff().fillna(0)
        df['speed_change'] = df.groupby(gcols)['s'].diff().fillna(0)
        df['direction_change'] = df.groupby(gcols)['dir'].diff().fillna(0)
        # Handle 360° wrap-around for direction changes
        df['direction_change'] = df['direction_change'].apply(
            lambda x: x if abs(x) < 180 else x - 360 * np.sign(x)
        )
    
    # ========================================================================
    # GROUP 6: FIELD POSITION FEATURES (4 features)
    # ========================================================================
    """
    SPATIAL CONTEXT: Field boundaries and strategic positioning
    
    Captures player position relative to field constraints:
    - dist_from_left/right: Distance from sidelines (0-53.3 yards)
    - dist_from_sideline: Minimum distance to nearest sideline
    - dist_from_endzone: Distance from nearest endzone (0-120 yards)
    
    Purpose: Encode field constraints that influence player movement patterns
    """
    df['dist_from_left'] = df['y']
    df['dist_from_right'] = 53.3 - df['y']
    df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
    df['dist_from_endzone'] = np.minimum(df['x'], 120 - df['x'])
    
    # ========================================================================
    # GROUP 7: ROLE-SPECIFIC FEATURES (3 features)
    # ========================================================================
    """
    PLAYER ROLE OPTIMIZATION: Position-specific performance metrics
    
    Tailors features to specific player responsibilities:
    - receiver_optimality: How well receivers align with ball trajectory
    - receiver_deviation: Lateral movement inefficiency for receivers
    - defender_closing_speed: Defensive pursuit effectiveness
    
    Purpose: Capture role-specific movement quality and tactical execution
    """
    if 'is_receiver' in df.columns and 'velocity_alignment' in df.columns:
        df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
        df['receiver_deviation'] = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0))
    if 'is_coverage' in df.columns and 'closing_speed' in df.columns:
        df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']
    
    # ========================================================================
    # GROUP 8: TIME FEATURES (2 features)
    # ========================================================================
    """
    TEMPORAL PROGRESSION: Play timeline and phase identification
    
    Captures where we are in the play timeline:
    - frames_elapsed: Absolute time since play start
    - normalized_time: Relative progress through play (0-1)
    
    Purpose: Enable time-aware predictions and phase-specific behavior modeling
    """
    df['frames_elapsed'] = df.groupby(gcols).cumcount()
    df['normalized_time'] = df.groupby(gcols)['frames_elapsed'].transform(
        lambda x: x / (x.max() + 1)
    )
    
    # ========================================================================
    # GROUP 9: JERK FEATURES (3 features)
    # ========================================================================
    """
    THIRD-ORDER DYNAMICS: Rate of acceleration change (jerk)
    
    Captures sudden changes in acceleration patterns:
    - jerk: Overall acceleration change rate
    - jerk_x/y: Directional components of jerk
    
    Purpose: Detect sudden movement decisions, cuts, and agility maneuvers
    """
    if 'a' in df.columns:
        df['jerk'] = df.groupby(gcols)['a'].diff().fillna(0) * 10.0  # FPS=10
    if 'acceleration_x' in df.columns and 'acceleration_y' in df.columns:
        df['jerk_x'] = df.groupby(gcols)['acceleration_x'].diff().fillna(0) * 10.0
        df['jerk_y'] = df.groupby(gcols)['acceleration_y'].diff().fillna(0) * 10.0
    
    # ========================================================================
    # GROUP 10: CURVATURE LAND FEATURES (8 features)
    # ========================================================================
    """
    TRAJECTORY CURVATURE ANALYSIS: Path geometry relative to ball landing
    
    Advanced geometric analysis of movement paths:
    - bearing_to_land_signed: Angular direction to ball landing point
    - land_lateral_offset: Perpendicular distance from trajectory to landing
    - curvature_signed/abs: Path curvature (positive = right turn)
    - curvature_roll3/5: Smoothed curvature over multiple frames
    
    Purpose: Capture route-running precision and path optimization strategies
    """
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        # Signed bearing to ball landing position
        dx = df['ball_land_x'] - df['x']
        dy = df['ball_land_y'] - df['y']
        bearing = np.arctan2(dy, dx)
        dir_rad = np.deg2rad(df['dir'].fillna(0))
        df['bearing_to_land_signed'] = np.rad2deg(np.arctan2(
            np.sin(bearing - dir_rad), np.cos(bearing - dir_rad)
        ))
        
        # Lateral offset from trajectory to landing point
        ux, uy = np.cos(dir_rad), np.sin(dir_rad)
        df['land_lateral_offset'] = dy * ux - dx * uy
        
        # Trajectory curvature analysis
        ddir = df.groupby(gcols)['dir'].diff().fillna(0)
        ddir = np.where(np.abs(ddir) > 180, ddir - 360 * np.sign(ddir), ddir)
        curvature = np.deg2rad(ddir) / (df['s'].replace(0, np.nan) * 0.1 + 1e-6)
        df['curvature_signed'] = curvature.fillna(0)
        df['curvature_abs'] = np.abs(df['curvature_signed'])
        
        # Rolling curvature averages
        for window in [3, 5]:
            df[f'curvature_roll{window}'] = df.groupby(gcols)['curvature_signed'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
    
    # ========================================================================
    # GROUP 11: ENHANCED LAG FEATURES (12 features)
    # ========================================================================
    """
    SHORT-TERM HISTORICAL CONTEXT: Recent positions and velocities
    
    Provides immediate historical context:
    - Lags: 1, 2, 3 frames (0.1s, 0.2s, 0.3s at 10 FPS)
    - Variables: x, y, velocity_x, velocity_y
    
    Purpose: Capture immediate movement history and short-term patterns
    Combined with GROUP 4, provides complete temporal context (0.1s - 0.5s)
    """
    for lag in [1, 2, 3]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(gcols)[col].shift(lag).fillna(0)
    
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
    print("Step 2/4: Adding advanced features...")
    input_df = add_advanced_features(input_df)
    
    # PLAYER INTERACTION FEATURES
    print("Step 3/4: Adding player interaction features...")
    use_players_interactions = True  # Enable player interaction features
    if use_players_interactions:
        agg_rows = []
        # Group once (avoid overhead of apply per small group)
        for (g, p, f), grp in input_df.groupby(['game_id', 'play_id', 'frame_id'], sort=False):
            n = len(grp)
            nfl_ids = grp['nfl_id'].to_numpy()
            # Only compute/emit for player_to_predict==True (if column exists)
            compute_mask = grp['player_to_predict'].to_numpy().astype(bool) if 'player_to_predict' in grp.columns else np.ones(n, dtype=bool)
            if n < 2:
                # Create empty stats rows (NaNs) only for players to predict
                for nid in nfl_ids[compute_mask]:
                    agg_rows.append({
                        'game_id': g, 'play_id': p, 'frame_id': f, 'nfl_id': nid,
                        'distance_to_player_mean_offense': np.nan,
                        'distance_to_player_min_offense': np.nan,
                        'distance_to_player_max_offense': np.nan,
                        'relative_velocity_magnitude_mean_offense': np.nan,
                        'relative_velocity_magnitude_min_offense': np.nan,
                        'relative_velocity_magnitude_max_offense': np.nan,
                        'angle_to_player_mean_offense': np.nan,
                        'angle_to_player_min_offense': np.nan,
                        'angle_to_player_max_offense': np.nan,
                        'distance_to_player_mean_defense': np.nan,
                        'distance_to_player_min_defense': np.nan,
                        'distance_to_player_max_defense': np.nan,
                        'relative_velocity_magnitude_mean_defense': np.nan,
                        'relative_velocity_magnitude_min_defense': np.nan,
                        'relative_velocity_magnitude_max_defense': np.nan,
                        'angle_to_player_mean_defense': np.nan,
                        'angle_to_player_min_defense': np.nan,
                        'angle_to_player_max_defense': np.nan,
                        'nearest_opponent_dist': np.nan,
                        'nearest_opponent_angle': np.nan,
                        'nearest_opponent_rel_speed': np.nan,
                    })
                continue

            x = grp['x'].to_numpy(dtype=np.float32)
            y = grp['y'].to_numpy(dtype=np.float32)
            vx = grp['velocity_x'].to_numpy(dtype=np.float32)
            vy = grp['velocity_y'].to_numpy(dtype=np.float32)
            is_offense = grp['is_offense'].to_numpy()
            is_defense = grp['is_defense'].to_numpy()

            # Pairwise deltas (broadcast)
            dx = x[None, :] - x[:, None]        # (n,n) x_j - x_i reversed later for angle
            dy = y[None, :] - y[:, None]
            # Angle from i -> j (want y_j - y_i, x_j - x_i)
            angle_mat = np.arctan2(-dy, -dx)    # because dx currently x[None]-x[:,None] => -(x_j - x_i)

            # Distances
            dist = np.sqrt(dx ** 2 + dy ** 2)
            # Relative velocity magnitudes
            dvx = vx[:, None] - vx[None, :]
            dvy = vy[:, None] - vy[None, :]
            rel_speed = np.sqrt(dvx ** 2 + dvy ** 2)

            # Offense mask (exclude self)
            offense_mask = (is_offense[:, None] == is_offense[None, :])
            np.fill_diagonal(offense_mask, False)

            # Defense mask (exclude self)
            defense_mask = (is_defense[:, None] == is_defense[None, :])
            np.fill_diagonal(defense_mask, False)

            # Opponent mask (exclude self)
            opp_mask = (is_offense[:, None] != is_offense[None, :])
            np.fill_diagonal(opp_mask, False)

            # Mask out self distances
            dist_diag_nan = dist.copy()
            np.fill_diagonal(dist_diag_nan, np.nan)
            rel_diag_nan = rel_speed.copy()
            np.fill_diagonal(rel_diag_nan, np.nan)
            angle_diag_nan = angle_mat.copy()
            np.fill_diagonal(angle_diag_nan, np.nan)

            def masked_stats(mat, mask):
                # mat, mask shape (n,n)
                masked = np.where(mask, mat, np.nan)
                cnt = mask.sum(axis=1)
                mean = np.nanmean(masked, axis=1)
                amin = np.nanmin(masked, axis=1)
                amax = np.nanmax(masked, axis=1)
                # Rows with zero valid -> set nan
                zero = cnt == 0
                mean[zero] = np.nan; amin[zero] = np.nan; amax[zero] = np.nan
                return mean, amin, amax

            d_mean_o, d_min_o, d_max_o = masked_stats(dist_diag_nan, offense_mask)
            v_mean_o, v_min_o, v_max_o = masked_stats(rel_diag_nan, offense_mask)
            a_mean_o, a_min_o, a_max_o = masked_stats(angle_diag_nan, offense_mask)

            d_mean_d, d_min_d, d_max_d = masked_stats(dist_diag_nan, defense_mask)
            v_mean_d, v_min_d, v_max_d = masked_stats(rel_diag_nan, defense_mask)
            a_mean_d, a_min_d, a_max_d = masked_stats(angle_diag_nan, defense_mask)

            # NEW: nearest opponent stats
            masked_dist_opp = np.where(opp_mask, dist_diag_nan, np.nan)         # (n,n)
            nearest_dist = np.nanmin(masked_dist_opp, axis=1)                   # (n,)
            nearest_idx = np.nanargmin(masked_dist_opp, axis=1)                 # (n,)
            # Guard where all-NaN rows (no opponents)
            all_nan = ~np.isfinite(nearest_dist)
            nearest_idx_safe = nearest_idx.copy()
            nearest_idx_safe[all_nan] = 0
            nearest_angle = np.take_along_axis(angle_diag_nan, nearest_idx_safe[:, None], axis=1).squeeze(1)
            nearest_rel = np.take_along_axis(rel_diag_nan, nearest_idx_safe[:, None], axis=1).squeeze(1)
            nearest_angle[all_nan] = np.nan
            nearest_rel[all_nan] = np.nan

            for idx, nid in enumerate(nfl_ids):
                if not compute_mask[idx]:
                    continue  # only for player_to_predict==True
                agg_rows.append({
                    'game_id': g, 'play_id': p, 'frame_id': f, 'nfl_id': nid,
                    'distance_to_player_mean_offense': d_mean_o[idx],
                    'distance_to_player_min_offense': d_min_o[idx],
                    'distance_to_player_max_offense': d_max_o[idx],
                    'relative_velocity_magnitude_mean_offense': v_mean_o[idx],  # Fixed typo: was v_mean_o[ix]
                    'relative_velocity_magnitude_min_offense': v_min_o[idx],
                    'relative_velocity_magnitude_max_offense': v_max_o[idx],
                    'angle_to_player_mean_offense': a_mean_o[idx],
                    'angle_to_player_min_offense': a_min_o[idx],
                    'angle_to_player_max_offense': a_max_o[idx],
                    'distance_to_player_mean_defense': d_mean_d[idx],
                    'distance_to_player_min_defense': d_min_d[idx],
                    'distance_to_player_max_defense': d_max_d[idx],
                    'relative_velocity_magnitude_mean_defense': v_mean_d[idx],
                    'relative_velocity_magnitude_min_defense': v_min_d[idx],
                    'relative_velocity_magnitude_max_defense': v_max_d[idx],
                    'angle_to_player_mean_defense': a_mean_d[idx],
                    'angle_to_player_min_defense': a_min_d[idx],
                    'angle_to_player_max_defense': a_max_d[idx],
                    'nearest_opponent_dist': nearest_dist[idx],
                    'nearest_opponent_angle': nearest_angle[idx],
                    'nearest_opponent_rel_speed': nearest_rel[idx],
                })

        interaction_agg = pd.DataFrame(agg_rows)
        input_df = input_df.merge(
            interaction_agg,
            on=['game_id', 'play_id', 'frame_id', 'nfl_id'],
            how='left'
        )
    else:
        print("Skipping player interaction feature computation (use_players_interactions=False).")
    
    # GNN LITE FEATURES
    print("Step 4/4: Adding GNN Lite features...")
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
        
        # PLAYER INTERACTION FEATURES (21)
        'd_mean_o', 'd_min_o', 'd_max_o',  # offensive distance stats
        'd_mean_d', 'd_min_d', 'd_max_d',  # defensive distance stats
        'v_mean_o', 'v_min_o', 'v_max_o',  # offensive velocity stats
        'v_mean_d', 'v_min_d', 'v_max_d',  # defensive velocity stats
        'a_mean_o', 'a_min_o', 'a_max_o',  # offensive angle stats
        'a_mean_d', 'a_min_d', 'a_max_d',  # defensive angle stats
        'nearest_opp_dist', 'nearest_opp_angle', 'nearest_opp_rel_speed',  # nearest opponent stats
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
class Seq2SeqGRU(nn.Module):
    """
    Dual-GRU Encoder-Decoder architecture adapted from my_gru.py
    Modified to work with the advanced feature pipeline
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, horizon=94):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Encoder GRU processes the input sequence
        self.encoder_gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Decoder GRU generates the output sequence step by step
        self.decoder_gru = nn.GRU(
            2,  # Only x, y coordinates for decoder input
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection layer
        self.out = nn.Linear(hidden_size, 2)  # Outputs dx, dy
        
        # Optional: Add attention mechanism for better performance
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads=4, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, teaching_force=True, targets=None):
        """
        Forward pass for Seq2Seq GRU
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            teaching_force: Whether to use teacher forcing during training
            targets: Target sequences for teacher forcing (batch_size, horizon, 2)
        """
        batch_size = x.size(0)
        
        # Encoder: Process input sequence
        encoder_outputs, hidden = self.encoder_gru(x)
        
        # Optional attention over encoder outputs
        if self.use_attention:
            # Use the last hidden state as query for attention
            last_hidden = hidden[-1].unsqueeze(1)  # (batch_size, 1, hidden_size)
            attended, _ = self.attention(
                last_hidden, encoder_outputs, encoder_outputs
            )
            # Combine attended output with hidden state
            hidden = hidden + attended.transpose(0, 1)  # Add residual connection
            hidden = self.attention_norm(hidden.transpose(0, 1)).transpose(0, 1)
        
        # Decoder: Generate output sequence step by step
        # Initialize decoder input with last position from encoder (or zeros)
        if x.size(-1) >= 2:  # If input contains x, y coordinates
            decoder_input = x[:, -1:, :2]  # Use last x, y as initial input
        else:
            decoder_input = torch.zeros(batch_size, 1, 2, device=x.device)
        
        outputs = []
        
        for t in range(self.horizon):
            # Decoder step
            decoder_output, hidden = self.decoder_gru(decoder_input, hidden)
            
            # Project to output space (dx, dy)
            pred = self.out(decoder_output)  # (batch_size, 1, 2)
            outputs.append(pred)
            
            # Prepare next input
            if teaching_force and targets is not None and t < targets.size(1) - 1:
                # Use ground truth for next input (teacher forcing)
                decoder_input = targets[:, t:t+1, :]
            else:
                # Use prediction for next input (autoregressive)
                decoder_input = pred
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # (batch_size, horizon, 2)
        
        # Convert from deltas to cumulative positions
        return torch.cumsum(outputs, dim=1)

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
    model = Seq2SeqGRU(input_dim, Config.HIDDEN_DIM, num_layers=2, horizon=horizon).to(device)
    
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
    
    # Save model if configured
    if Config.SAVE_ARTIFACTS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Config.MODEL_DIR / f"seq2seq_gru_model_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': Config.HIDDEN_DIM,
            'horizon': horizon,
            'best_loss': best_loss,
            'config': {
                'WINDOW_SIZE': Config.WINDOW_SIZE,
                'HIDDEN_DIM': Config.HIDDEN_DIM,
                'LEARNING_RATE': Config.LEARNING_RATE,
                'BATCH_SIZE': Config.BATCH_SIZE
            }
        }, model_path)
        print(f"Model saved to: {model_path}")
    
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
