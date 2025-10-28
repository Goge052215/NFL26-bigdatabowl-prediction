# -------------------------------
# Global imports + cuDF accelerator
# -------------------------------
import os
USE_CUDF = False
try:
    # zero/low-code GPU acceleration for DataFrame ops
    os.environ["CUDF_PANDAS_BACKEND"] = "cudf"
    import pandas as pd
    import numpy as np
    import cupy as cp  # optional (not strictly required below)
    USE_CUDF = True
    print("using cuda_backend pandas for faster parallel data processing")
except Exception:
    print("cuda df not used")
    import pandas as pd
    import numpy as np

import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Constants & helpers
# -------------------------------
YARDS_TO_METERS = 0.9144
FPS = 10.0 
FIELD_LENGTH, FIELD_WIDTH = 120.0, 53.3

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
print("environment set up!")
def wrap_angle_deg(s):
    # map to (-180, 180]
    return ((s + 180.0) % 360.0) - 180.0

def unify_left_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror rightward plays so all samples are 'left' oriented (x,y, dir, o, ball_land)."""
    if 'play_direction' not in df.columns:
        return df
    df = df.copy()
    right = df['play_direction'].eq('right')
    # positions
    if 'x' in df.columns: df.loc[right, 'x'] = FIELD_LENGTH - df.loc[right, 'x']
    if 'y' in df.columns: df.loc[right, 'y'] = FIELD_WIDTH  - df.loc[right, 'y']
    # angles in degrees
    for col in ('dir','o'):
        if col in df.columns:
            df.loc[right, col] = (df.loc[right, col] + 180.0) % 360.0
    # ball landing
    if 'ball_land_x' in df.columns:
        df.loc[right, 'ball_land_x'] = FIELD_LENGTH - df.loc[right, 'ball_land_x']
    if 'ball_land_y' in df.columns:
        df.loc[right, 'ball_land_y'] = FIELD_WIDTH  - df.loc[right, 'ball_land_y']
    return df

def invert_to_original_direction(x_u, y_u, play_dir_right: bool):
    """Invert unified (left) coordinates back to original play direction."""
    if not play_dir_right:
        return float(x_u), float(y_u)
    return float(FIELD_LENGTH - x_u), float(FIELD_WIDTH - y_u)

# -------------------------------
# Config
# -------------------------------
class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    OUTPUT_DIR = Path("./outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

    SEED = 42
    N_FOLDS = 4
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3

    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94  # 不要改动这个！！！

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(Config.SEED)

# -------------------------------
# Sequence builder (unified frame + safe targets)
# -------------------------------
def build_play_direction_map(df_in: pd.DataFrame) -> pd.Series:
    """
    Return a Series indexed by (game_id, play_id) with values 'left'/'right'.
    This keeps a clean MultiIndex that works for both pandas and cuDF pandas-API.
    """
    s = (
        df_in[['game_id','play_id','play_direction']]
        .drop_duplicates()
        .set_index(['game_id','play_id'])['play_direction']
    )
    return s  # MultiIndex Series


def apply_direction_to_df(df: pd.DataFrame, dir_map: pd.Series) -> pd.DataFrame:
    """
    Attach play_direction (if missing) and then unify to 'left'.
    dir_map must be the MultiIndex Series produced by build_play_direction_map.
    """
    if 'play_direction' not in df.columns:
        dir_df = dir_map.reset_index()  # -> columns: game_id, play_id, play_direction
        df = df.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
    return unify_left_direction(df)

def prepare_sequences_with_advanced_features(
        input_df, output_df=None, test_template=None, 
        is_training=True, window_size=10, feature_groups=None):

    print(f"\n{'='*80}")
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES (UNIFIED FRAME)")
    print(f"{'='*80}")
    print(f"Window size: {window_size}")

    if feature_groups is None:
        feature_groups = [
            'distance_rate','target_alignment','multi_window_rolling','extended_lags',
            'velocity_changes','field_position','role_specific','time_features','jerk_features',
            'player_interaction_distance',
        ]

    # Direction map and unify
    dir_map = build_play_direction_map(input_df)
    input_df_u = unify_left_direction(input_df)
    
    if is_training:
        out_u = apply_direction_to_df(output_df, dir_map)
        target_rows = out_u
        target_groups = out_u[['game_id','play_id','nfl_id']].drop_duplicates()
    else:
        if 'play_direction' not in test_template.columns:
            dir_df = dir_map.reset_index()
            test_template = test_template.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
        target_rows = test_template
        target_groups = target_rows[['game_id','play_id','nfl_id','play_direction']].drop_duplicates()
        
    assert target_rows[['game_id','play_id','play_direction']].isna().sum().sum() == 0, \
        "play_direction merge failed; check (game_id, play_id) coverage"
    print("play_direction merge OK:", target_rows['play_direction'].value_counts(dropna=False).to_dict())

    # --- FE ---
    fe = FeatureEngineer(feature_groups)
    processed_df, feature_cols = fe.transform(input_df_u)

    # --- Build sequences (OPTIMIZED) ---
    print("\nStep 3/3: Creating sequences...")
    
    # 优化1: 预先排序并分组（避免重复set_index）
    processed_df = processed_df.sort_values(['game_id','play_id','nfl_id','frame_id'])
    
    # 优化2: 使用字典缓存分组数据（避免重复get_group）
    grouped_dict = {
        key: group[feature_cols].values 
        for key, group in processed_df.groupby(['game_id','play_id','nfl_id'])
    }
    
    # 优化3: 预计算分组统计（用于fillna）
    group_means = processed_df.groupby(['game_id','play_id','nfl_id'])[feature_cols].mean()
    
    # 预先获取x,y索引
    idx_x = feature_cols.index('x')
    idx_y = feature_cols.index('y')
    idx_fid = processed_df.columns.get_loc('frame_id')
    
    # 预分配列表（减少动态扩容）
    n_targets = len(target_groups)
    sequences = []
    targets_dx = [] if is_training else None
    targets_dy = [] if is_training else None
    targets_fids = [] if is_training else None
    seq_meta = []
    
    # 优化4: 转为numpy数组加速迭代
    target_array = target_groups.values
    
    # 优化5: 预处理training目标数据（避免重复筛选）
    if is_training:
        # 创建多级索引快速查找
        target_rows = target_rows.sort_values(['game_id','play_id','nfl_id','frame_id'])
        target_lookup = {
            (gid, pid, nid): group[['x','y','frame_id']].values
            for (gid, pid, nid), group in target_rows.groupby(['game_id','play_id','nfl_id'])
        }
    
    # 优化6: 批量处理（减少tqdm开销）
    for i in tqdm(range(n_targets), desc="Creating sequences"):
        if is_training:
            gid, pid, nid = target_array[i, :3]
            play_dir = None
        else:
            gid, pid, nid, play_dir = target_array[i, :4]
        
        key = (gid, pid, nid)
        
        # 快速查找
        if key not in grouped_dict:
            continue
        
        group_data = grouped_dict[key]
        
        # 获取窗口
        if len(group_data) >= window_size:
            input_window = group_data[-window_size:]
        else:
            if is_training:
                continue
            # 快速padding（使用numpy）
            pad_len = window_size - len(group_data)
            pad_array = np.full((pad_len, len(feature_cols)), np.nan, dtype=np.float32)
            input_window = np.vstack([pad_array, group_data])
        
        # 优化7: 向量化fillna（使用预计算的均值）
        if key in group_means.index:
            mean_vals = group_means.loc[key].values
            nan_mask = np.isnan(input_window)
            input_window = np.where(nan_mask, mean_vals, input_window)
        
        # 最终NaN处理
        if np.isnan(input_window).any():
            if is_training:
                continue
            input_window = np.nan_to_num(input_window, nan=0.0)
        
        sequences.append(input_window.astype(np.float32))
        
        # Training targets
        if is_training:
            if key not in target_lookup:
                continue
            
            target_data = target_lookup[key]  # [n_frames, 3] -> [x, y, frame_id]
            
            last_x = input_window[-1, idx_x]
            last_y = input_window[-1, idx_y]
            
            dx = (target_data[:, 0] - last_x).astype(np.float32)
            dy = (target_data[:, 1] - last_y).astype(np.float32)
            fids = target_data[:, 2].astype(np.int32)
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_fids.append(fids)
        
        # Metadata（优化：减少字典创建开销）
        seq_meta.append({
            'game_id': int(gid),
            'play_id': int(pid),
            'nfl_id': int(nid),
            'frame_id': int(processed_df[
                (processed_df['game_id']==gid) & 
                (processed_df['play_id']==pid) & 
                (processed_df['nfl_id']==nid)
            ]['frame_id'].iloc[-1]),
            'play_direction': play_dir,
        })
    
    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")

    if is_training:
        return sequences, targets_dx, targets_dy, targets_fids, seq_meta, feature_cols, dir_map
    return sequences, seq_meta, feature_cols, dir_map

# -------------------------------
# Feature Engineering
# -------------------------------
class FeatureEngineer:
    """
    Modular, ablation-friendly feature builder (pandas or cuDF pandas-API).
    """
    def __init__(self, feature_groups_to_create):
        self.gcols = ['game_id', 'play_id', 'nfl_id']
        self.active_groups = feature_groups_to_create
        self.feature_creators = {
            'distance_rate': self._create_distance_rate_features,
            'target_alignment': self._create_target_alignment_features,
            'multi_window_rolling': self._create_multi_window_rolling_features,
            'extended_lags': self._create_extended_lag_features,
            'velocity_changes': self._create_velocity_change_features,
            'field_position': self._create_field_position_features,
            'role_specific': self._create_role_specific_features,
            'time_features': self._create_time_features,
            'jerk_features': self._create_jerk_features,
            'curvature_land_features': self._create_curvature_land_features,
            'player_interaction_distance': self._create_player_interaction_distance_features,
        }
        self.created_feature_cols = []

    def _height_to_feet(self, height_str):
        try:
            ft, inches = map(int, str(height_str).split('-'))
            return ft + inches / 12
        except Exception:
            return 6.0

    def _create_basic_features(self, df):
        print("Step 1/3: Adding basic features...")
        df = df.copy()
        df['player_height_feet'] = df['player_height'].apply(self._height_to_feet)

        # Correct kinematics: dir is from +x CCW
        dir_rad = np.deg2rad(df['dir'].fillna(0.0).astype('float32'))
        df['velocity_x']     = df['s'] * np.cos(dir_rad)
        df['velocity_y']     = df['s'] * np.sin(dir_rad)
        df['acceleration_x'] = df['a'] * np.cos(dir_rad)
        df['acceleration_y'] = df['a'] * np.sin(dir_rad)

        # Roles
        df['is_offense']  = (df['player_side'] == 'Offense').astype(np.int8)
        df['is_defense']  = (df['player_side'] == 'Defense').astype(np.int8)
        df['is_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(np.int8)
        df['is_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(np.int8)
        df['is_passer']   = (df['player_role'] == 'Passer').astype(np.int8)

        # Energetics (consistent units)
        mass_kg = df['player_weight'].fillna(200.0) / 2.20462
        v_ms = df['s'] * YARDS_TO_METERS
        df['momentum_x'] = mass_kg * df['velocity_x'] * YARDS_TO_METERS
        df['momentum_y'] = mass_kg * df['velocity_y'] * YARDS_TO_METERS
        df['kinetic_energy'] = 0.5 * mass_kg * (v_ms ** 2)

        # Ball landing geometry (static)
        if {'ball_land_x','ball_land_y'}.issubset(df.columns):
            ball_dx = df['ball_land_x'] - df['x']
            ball_dy = df['ball_land_y'] - df['y']
            dist = np.hypot(ball_dx, ball_dy)
            df['distance_to_ball'] = dist
            inv = 1.0 / (dist + 1e-6)
            df['ball_direction_x'] = ball_dx * inv
            df['ball_direction_y'] = ball_dy * inv
            df['closing_speed'] = (
                df['velocity_x'] * df['ball_direction_x'] +
                df['velocity_y'] * df['ball_direction_y']
            )

        base = [
            'x','y','s','a','o','dir','frame_id','ball_land_x','ball_land_y',
            'player_height_feet','player_weight',
            'velocity_x','velocity_y','acceleration_x','acceleration_y',
            'momentum_x','momentum_y','kinetic_energy',
            'is_offense','is_defense','is_receiver','is_coverage','is_passer',
            'distance_to_ball','ball_direction_x','ball_direction_y','closing_speed'
        ]
        self.created_feature_cols.extend([c for c in base if c in df.columns])
        return df

    # ---- feature groups ----
    def _create_distance_rate_features(self, df):
        new_cols = []
        if 'distance_to_ball' in df.columns:
            d = df.groupby(self.gcols)['distance_to_ball'].diff()
            df['d2ball_dt']  = d.fillna(0.0) * FPS
            df['d2ball_ddt'] = df.groupby(self.gcols)['d2ball_dt'].diff().fillna(0.0) * FPS
            df['time_to_intercept'] = (df['distance_to_ball'] /
                                       (df['d2ball_dt'].abs() + 1e-3)).clip(0, 10)
            new_cols = ['d2ball_dt','d2ball_ddt','time_to_intercept']
        return df, new_cols

    def _create_target_alignment_features(self, df):
        new_cols = []
        if {'ball_direction_x','ball_direction_y','velocity_x','velocity_y'}.issubset(df.columns):
            df['velocity_alignment'] = df['velocity_x']*df['ball_direction_x'] + df['velocity_y']*df['ball_direction_y']
            df['velocity_perpendicular'] = df['velocity_x']*(-df['ball_direction_y']) + df['velocity_y']*df['ball_direction_x']
            new_cols.extend(['velocity_alignment','velocity_perpendicular'])
            if {'acceleration_x','acceleration_y'}.issubset(df.columns):
                df['accel_alignment'] = df['acceleration_x']*df['ball_direction_x'] + df['acceleration_y']*df['ball_direction_y']
                new_cols.append('accel_alignment')
        return df, new_cols

    def _create_multi_window_rolling_features(self, df):
        # keep it simple & compatible (works with cuDF pandas-API); vectorized rolling per group
        new_cols = []
        for window in (3, 5, 10):
            for col in ('velocity_x','velocity_y','s','a'):
                if col in df.columns:
                    r_mean = df.groupby(self.gcols)[col].rolling(window, min_periods=1).mean()
                    r_std  = df.groupby(self.gcols)[col].rolling(window, min_periods=1).std()
                    # align indices
                    r_mean = r_mean.reset_index(level=list(range(len(self.gcols))), drop=True)
                    r_std  = r_std.reset_index(level=list(range(len(self.gcols))), drop=True)
                    df[f'{col}_roll{window}'] = r_mean
                    df[f'{col}_std{window}']  = r_std.fillna(0.0)
                    new_cols.extend([f'{col}_roll{window}', f'{col}_std{window}'])
        return df, new_cols

    def _create_extended_lag_features(self, df):
        new_cols = []
        for lag in (1,2,3,4,5):
            for col in ('x','y','velocity_x','velocity_y'):
                if col in df.columns:
                    g = df.groupby(self.gcols)[col]
                    lagv = g.shift(lag)
                    # safe fill for first frames (no "future" leakage)
                    df[f'{col}_lag{lag}'] = lagv.fillna(g.transform('first'))
                    new_cols.append(f'{col}_lag{lag}')
        return df, new_cols

    def _create_velocity_change_features(self, df):
        new_cols = []
        if 'velocity_x' in df.columns:
            df['velocity_x_change'] = df.groupby(self.gcols)['velocity_x'].diff().fillna(0.0)
            df['velocity_y_change'] = df.groupby(self.gcols)['velocity_y'].diff().fillna(0.0)
            df['speed_change']      = df.groupby(self.gcols)['s'].diff().fillna(0.0)
            d = df.groupby(self.gcols)['dir'].diff().fillna(0.0)
            df['direction_change']  = wrap_angle_deg(d)
            new_cols = ['velocity_x_change','velocity_y_change','speed_change','direction_change']
        return df, new_cols

    def _create_field_position_features(self, df):
        df['dist_from_left'] = df['y']
        df['dist_from_right'] = FIELD_WIDTH - df['y']
        df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
        df['dist_from_endzone']  = np.minimum(df['x'], FIELD_LENGTH - df['x'])
        return df, ['dist_from_sideline','dist_from_endzone']

    def _create_role_specific_features(self, df):
        new_cols = []
        if {'is_receiver','velocity_alignment'}.issubset(df.columns):
            df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
            df['receiver_deviation']  = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0.0))
            new_cols.extend(['receiver_optimality','receiver_deviation'])
        if {'is_coverage','closing_speed'}.issubset(df.columns):
            df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']
            new_cols.append('defender_closing_speed')
        return df, new_cols

    def _create_time_features(self, df):
        df['frames_elapsed']  = df.groupby(self.gcols).cumcount()
        df['normalized_time'] = df.groupby(self.gcols)['frames_elapsed'].transform(
            lambda x: x / (x.max() + 1e-9)
        )
        return df, ['frames_elapsed','normalized_time']

    def _create_jerk_features(self, df):
        new_cols = []
        if 'a' in df.columns:
            df['jerk'] = df.groupby(self.gcols)['a'].diff().fillna(0.0) * FPS
            new_cols.append('jerk')
        if {'acceleration_x','acceleration_y'}.issubset(df.columns):
            df['jerk_x'] = df.groupby(self.gcols)['acceleration_x'].diff().fillna(0.0) * FPS
            df['jerk_y'] = df.groupby(self.gcols)['acceleration_y'].diff().fillna(0.0) * FPS
            new_cols.extend(['jerk_x','jerk_y'])
        return df, new_cols
        
    def _create_curvature_land_features(self, df):
        if {'ball_land_x','ball_land_y'}.issubset(df.columns):
            dx = df['ball_land_x'] - df['x']
            dy = df['ball_land_y'] - df['y']
            bearing = np.arctan2(dy, dx)
            a_dir = np.deg2rad(df['dir'].fillna(0.0).values)
            # 有符号方位差
            df['bearing_to_land_signed'] = np.rad2deg(np.arctan2(np.sin(bearing - a_dir), np.cos(bearing - a_dir)))
            # 侧向偏差：d × u (2D cross, z 分量)
            ux, uy = np.cos(a_dir), np.sin(a_dir)
            df['land_lateral_offset'] = dy*ux - dx*uy  # >0 落点在左侧
    
        # 曲率（按序列）
        ddir = df.groupby(self.gcols)['dir'].diff().fillna(0.0)
        ddir = ((ddir + 180.0) % 360.0) - 180.0
        curvature = np.deg2rad(ddir).astype('float32') / (df['s'].replace(0, np.nan).astype('float32') * 0.1 + 1e-6)
        df['curvature_signed'] = curvature.fillna(0.0)
        df['curvature_abs'] = df['curvature_signed'].abs()
    
        # 窗口均值（3/5）
        for w in (3,5):
            r = df.groupby(self.gcols)['curvature_signed'].rolling(w, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
            df[f'curv_signed_roll{w}'] = r
            r2 = df.groupby(self.gcols)['curvature_abs'].rolling(w, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
            df[f'curv_abs_roll{w}'] = r2
    
        new_cols = ['bearing_to_land_signed','land_lateral_offset',
                    'curvature_signed','curvature_abs','curv_signed_roll3','curv_abs_roll3',
                    'curv_signed_roll5','curv_abs_roll5']
        return df, [c for c in new_cols if c in df.columns]
        
    def _create_player_interaction_distance_features(self, df):
        new_cols = []
        
        # 确保必要列存在
        if not {'x', 'y', 'velocity_x', 'velocity_y', 'is_offense'}.issubset(df.columns):
            return df, new_cols
        
        # 按play分组计算
        grouped_features = []
        for (gid, pid, fid), frame_df in df.groupby(['game_id', 'play_id', 'frame_id']):
            frame_df = frame_df.copy()
            
            # 提取坐标和速度（float32降低内存）
            positions = frame_df[['x', 'y']].values.astype(np.float32)
            velocities = frame_df[['velocity_x', 'velocity_y']].values.astype(np.float32)
            is_offense = frame_df['is_offense'].values
            
            n_players = len(frame_df)
            
            # 预计算距离矩阵（使用scipy优化）
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(positions, positions, metric='euclidean').astype(np.float32)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # 预计算队友/对手掩码
            is_opponent_matrix = is_offense[:, np.newaxis] != is_offense[np.newaxis, :]
            is_teammate_matrix = (is_offense[:, np.newaxis] == is_offense[np.newaxis, :]) & (np.arange(n_players)[:, None] != np.arange(n_players))
            
            # 向量化计算最小距离
            opponent_dists = np.where(is_opponent_matrix, dist_matrix, np.inf)
            teammate_dists = np.where(is_teammate_matrix, dist_matrix, np.inf)
            
            min_opponent_dist = np.minimum(np.min(opponent_dists, axis=1), 999.0)
            min_teammate_dist = np.minimum(np.min(teammate_dists, axis=1), 999.0)
            
            # 向量化密度计算（5码内）
            opponent_density = np.sum((opponent_dists < 5.0), axis=1).astype(np.int8)
            teammate_density = np.sum((teammate_dists < 5.0), axis=1).astype(np.int8)
            
            # 简化相对速度计算（使用近似）
            nearest_opponent_idx = np.argmin(opponent_dists, axis=1)
            valid_mask = min_opponent_dist < 20.0
            
            # 向量化速度计算
            vel_diff = velocities - velocities[nearest_opponent_idx]  # (n, 2)
            pos_diff = positions[nearest_opponent_idx] - positions  # (n, 2)
            
            # 点积投影（向量化）
            relative_velocity = np.sum(vel_diff * pos_diff, axis=1) / (min_opponent_dist + 1e-6)
            relative_velocity = np.where(valid_mask, relative_velocity, 0.0).astype(np.float32)
            
            # 批量赋值
            frame_df['min_opponent_distance'] = min_opponent_dist
            frame_df['min_teammate_distance'] = min_teammate_dist
            frame_df['relative_velocity_to_nearest_opponent'] = relative_velocity
            frame_df['opponent_density_5yd'] = opponent_density
            frame_df['teammate_density_5yd'] = teammate_density
            
            grouped_features.append(frame_df)
        
        # 合并结果
        result_df = pd.concat(grouped_features, ignore_index=True)
        
        new_cols = [
            'min_opponent_distance',
            'min_teammate_distance', 
            'relative_velocity_to_nearest_opponent',
            'opponent_density_5yd',
            'teammate_density_5yd'
        ]
        
        return result_df, new_cols

    def transform(self, df):
        df = df.copy().sort_values(['game_id','play_id','nfl_id','frame_id'])
        df = self._create_basic_features(df)

        print("\nStep 2/3: Adding selected advanced features...")
        for group_name in self.active_groups:
            if group_name in self.feature_creators:
                creator = self.feature_creators[group_name]
                df, new_cols = creator(df)
                self.created_feature_cols.extend(new_cols)
                print(f"  [+] Added '{group_name}' ({len(new_cols)} cols)")
            else:
                print(f"  [!] Unknown feature group: {group_name}")

        final_cols = sorted(set(self.created_feature_cols))
        print(f"\nTotal features created: {len(final_cols)}")
        return df, final_cols

from torch.cuda.amp import autocast, GradScaler

class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03, velocity_penalty_weight=0.01,
                 acceleration_penalty_weight=0.0, use_huber_for_penalty=False):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.velocity_penalty_weight = velocity_penalty_weight
        self.acceleration_penalty_weight = acceleration_penalty_weight
        self.use_huber_for_penalty = use_huber_for_penalty
        
        # 预计算标志位（避免每次forward都判断）
        self.has_velocity_penalty = velocity_penalty_weight > 0
        self.has_acceleration_penalty = acceleration_penalty_weight > 0
        self.has_time_decay = time_decay > 0
        
        # 缓存时间衰减权重（对于固定长度序列）
        self._cached_time_weights = {}
    
    def _get_time_weights(self, length, device):
        """缓存时间衰减权重以避免重复计算"""
        key = (length, device)
        if key not in self._cached_time_weights:
            t = torch.arange(length, device=device, dtype=torch.float32)
            self._cached_time_weights[key] = torch.exp(-self.time_decay * t).view(1, -1)
        return self._cached_time_weights[key]
    
    def _huber_loss(self, err):
        """统一的Huber损失计算"""
        abs_err = torch.abs(err)
        return torch.where(abs_err <= self.delta,
                          0.5 * err.square(),  # 使用.square()代替 * 运算
                          self.delta * (abs_err - 0.5 * self.delta))
    
    def forward(self, pred, target, mask):
        err = pred - target
        
        # ===== 主Huber损失 =====
        huber = self._huber_loss(err)
        
        # 时间衰减权重（一次性计算，复用于所有项）
        if self.has_time_decay:
            time_weights = self._get_time_weights(pred.size(1), pred.device)
            mask_weighted = mask * time_weights
            huber = huber * time_weights
        else:
            mask_weighted = mask
            time_weights = None
        
        main_loss = (huber * mask_weighted).sum() / (mask_weighted.sum() + 1e-8)
        
        # 早期退出：如果没有正则项，直接返回
        if not (self.has_velocity_penalty or self.has_acceleration_penalty):
            return main_loss
        
        total_loss = main_loss
        
        # ===== 速度平滑正则项 =====
        if self.has_velocity_penalty and pred.size(1) > 1:
            # 一阶差分（速度变化）
            velocity_diff = pred[:, 1:] - pred[:, :-1]
            mask_vel = mask[:, 1:]
            
            # 选择损失函数（减少分支）
            if self.use_huber_for_penalty:
                vel_loss = self._huber_loss(velocity_diff)
            else:
                vel_loss = velocity_diff.square()
            
            # 应用时间衰减（复用已计算的权重）
            if self.has_time_decay:
                time_weights_vel = time_weights[:, 1:]  # 切片比重新计算快
                vel_loss = vel_loss * time_weights_vel
                mask_vel = mask_vel * time_weights_vel
            
            velocity_penalty = (vel_loss * mask_vel).sum() / (mask_vel.sum() + 1e-8)
            total_loss = total_loss + self.velocity_penalty_weight * velocity_penalty
        
        # ===== 加速度平滑正则项 =====
        if self.has_acceleration_penalty and pred.size(1) > 2:
            # 二阶差分（加速度变化）- 复用velocity_diff避免重复计算
            if not self.has_velocity_penalty:
                velocity_diff = pred[:, 1:] - pred[:, :-1]
            
            acceleration = velocity_diff[:, 1:] - velocity_diff[:, :-1]
            mask_acc = mask[:, 2:]
            
            # 选择损失函数
            if self.use_huber_for_penalty:
                acc_loss = self._huber_loss(acceleration)
            else:
                acc_loss = acceleration.square()
            
            # 应用时间衰减
            if self.has_time_decay:
                time_weights_acc = time_weights[:, 2:]
                acc_loss = acc_loss * time_weights_acc
                mask_acc = mask_acc * time_weights_acc
            
            acceleration_penalty = (acc_loss * mask_acc).sum() / (mask_acc.sum() + 1e-8)
            total_loss = total_loss + self.acceleration_penalty_weight * acceleration_penalty
        
        return total_loss

class SeqModel(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        # 投影到可被num_heads整除的维度
        self.hidden_dim = 128
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # 位置编码（假设序列长度最大为10）- 注册为buffer避免梯度计算
        pos_encoding = torch.randn(1, 10, self.hidden_dim) * 0.02
        self.register_buffer('pos_encoding', pos_encoding)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 简化Pooling层 - 合并LayerNorm到注意力后
        self.pool_attn = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pool_ln = nn.LayerNorm(self.hidden_dim)  # 移到注意力后
        
        # 输出头 - 合并部分操作
        self.head_linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.head_prelu = nn.PReLU(num_parameters=self.hidden_dim)
        self.head_dropout = nn.Dropout(0.2)
        self.head_linear2 = nn.Linear(self.hidden_dim, horizon)
        
        # 预分配pool_query以避免每次forward都expand
        self._pool_query_cache = {}
    
    def _get_pool_query(self, batch_size):
        """缓存不同batch_size的pool_query"""
        if batch_size not in self._pool_query_cache:
            self._pool_query_cache[batch_size] = self.pool_query.expand(batch_size, -1, -1)
        return self._pool_query_cache[batch_size]
    
    def forward(self, x):
        # x: (B, seq_len, input_dim)
        B, seq_len, _ = x.shape
        
        # 投影输入 + 位置编码（合并操作）
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        h = self.transformer(x)  # (B, seq_len, hidden_dim)
        
        # 注意力池化（优化版本）
        q = self._get_pool_query(B)  # 复用缓存
        ctx, _ = self.pool_attn(q, h, h)  # 直接对h操作，不预先LayerNorm
        ctx = self.pool_ln(ctx.squeeze(1))  # LayerNorm放到注意力后，同时squeeze
        
        # 预测（展开Sequential避免列表迭代）
        out = self.head_linear1(ctx)
        out = self.head_prelu(out)
        out = self.head_dropout(out)
        out = self.head_linear2(out)  # (B, horizon)
        
        # 累积和（使用inplace操作不可行，保持原样）
        return torch.cumsum(out, dim=1)  # (B, horizon)

def prepare_targets(batch_axis, max_h):
    tensors, masks = [], []
    for arr in batch_axis:
        L = len(arr)
        padded = np.pad(arr, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        tensors.append(torch.tensor(padded))
        masks.append(torch.tensor(mask))
    return torch.stack(tensors), torch.stack(masks)

def train_model(X_train, y_train, X_val, y_val, input_dim, horizon, config):
    device = config.DEVICE
    model = SeqModel(input_dim, horizon).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    # 初始化混合精度训练
    scaler = GradScaler()

    # build batches (keep numpy → torch)
    def build_batches(X, Y):
        batches = []
        B = config.BATCH_SIZE
        for i in range(0, len(X), B):
            end = min(i + B, len(X))
            xs = torch.tensor(np.stack(X[i:end]).astype(np.float32))
            ys, ms = prepare_targets([Y[j] for j in range(i, end)], horizon)
            batches.append((xs, ys, ms))
        return batches

    tr_batches = build_batches(X_train, y_train)
    va_batches = build_batches(X_val,   y_val)

    best_loss, best_state, bad = float('inf'), None, 0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in tr_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                pred = model(bx)
                loss = criterion(pred, by, bm)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in va_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                
                # 验证阶段混合精度
                with autocast():
                    pred = model(bx)
                    val_losses.append(criterion(pred, by, bm).item())

        trl, val = float(np.mean(train_losses)), float(np.mean(val_losses))
        scheduler.step(val)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={trl:.4f}, val={val:.4f}")

        if val < best_loss:
            best_loss, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss

# ------------------------------_
# Main pipeline (MODIFICADO PARA ENSEMBLE DE SEMILLAS)
# ------------------------------_
class CFG(Config):
    # Añadimos la lista de semillas para el ensemble
    SEEDS = [7] # ¡Puedes cambiar o añadir más semillas aquí!

def main():
    cfg = CFG()
    print("="*80)
    print(f"PASO 2: PIPELINE MEJORADO CON ENSEMBLE DE {len(cfg.SEEDS)} SEMILLAS")
    print("="*80)
    print(f"Semillas a utilizar: {cfg.SEEDS}")
    print(f"cuDF backend activo? {USE_CUDF}")

    # [1/4] Carga de datos (se hace una sola vez)
    print("\n[1/4] Cargando datos...")
    train_input_files  = [cfg.DATA_DIR / f"train/input_2023_w{w:02d}.csv"  for w in range(1, 19)]
    train_output_files = [cfg.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
    train_input  = pd.concat([pd.read_csv(f) for f in train_input_files  if f.exists()], ignore_index=True)
    train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()], ignore_index=True)
    test_input     = pd.read_csv(cfg.DATA_DIR / "test_input.csv")
    test_template  = pd.read_csv(cfg.DATA_DIR / "test.csv")

    # [2/4] Preparación de secuencias (se hace una sola vez)
    print("\n[2/4] Construyendo secuencias con características AVANZADAS...")
    seqs, tdx, tdy, tfids, seq_meta, feat_cols, dir_map = prepare_sequences_with_advanced_features(
        train_input, output_df=train_output, is_training=True,
        window_size=cfg.WINDOW_SIZE
    )

    # numpy object arrays a listas para un manejo más fácil
    sequences = list(seqs)
    targets_dx = list(tdx)
    targets_dy = list(tdy)

    # [3/4] Entrenamiento con GroupKFold sobre múltiples semillas
    print("\n[3/4] Iniciando entrenamiento de ensemble...")
    
    # Contenedores para todos los modelos y escaladores de todas las ejecuciones
    all_models_x, all_models_y, all_scalers = [], [], []
    fold_rmse_list = []  # Para almacenar RMSE de cada fold
    
    groups = np.array([d['game_id'] for d in seq_meta])
    
    for seed in cfg.SEEDS:
        print(f"\n{'='*70}\n   Entrenando con Semilla (Seed): {seed}\n{'='*70}")
        set_seed(seed) # <--- ¡IMPORTANTE! Establecer la semilla para esta ejecución
        
        gkf = GroupKFold(n_splits=cfg.N_FOLDS)

        for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
            print(f"\n{'-'*60}\nFold {fold}/{cfg.N_FOLDS} para la semilla {seed}\n{'-'*60}")

            X_tr = [sequences[i] for i in tr]
            X_va = [sequences[i] for i in va]

            # Estandarización por fold
            scaler = StandardScaler()
            scaler.fit(np.vstack([s for s in X_tr]))

            X_tr_sc = np.stack([scaler.transform(s) for s in X_tr]).astype(np.float32)
            X_va_sc = np.stack([scaler.transform(s) for s in X_va]).astype(np.float32)

            # Entrenar modelo para X
            print("Entrenando modelo ΔX...")
            mx, loss_x = train_model(
                X_tr_sc, [targets_dx[i] for i in tr],
                X_va_sc, [targets_dx[i] for i in va],
                X_tr_sc.shape[-1], cfg.MAX_FUTURE_HORIZON, cfg
          )

            # Entrenar modelo para Y
            print("Entrenando modelo ΔY...")
            my, loss_y = train_model(
                X_tr_sc, [targets_dy[i] for i in tr],
                X_va_sc, [targets_dy[i] for i in va],
                X_tr_sc.shape[-1], cfg.MAX_FUTURE_HORIZON, cfg
            )
            
            # Guardar los modelos y el escalador de este fold
            all_models_x.append(mx)
            all_models_y.append(my)
            all_scalers.append(scaler)
            
            # Calcular RMSE en el conjunto de validación
            mx.eval()
            my.eval()
            with torch.no_grad():
                X_va_t = torch.tensor(X_va_sc).to(cfg.DEVICE)
                pred_dx = mx(X_va_t).cpu().numpy()
                pred_dy = my(X_va_t).cpu().numpy()
            
            # Preparar targets de validación para RMSE
            y_va_dx = [targets_dx[i] for i in va]
            y_va_dy = [targets_dy[i] for i in va]
            
            # Calcular RMSE: sqrt(mean((x_pred - x_true)^2 + (y_pred - y_true)^2))
            squared_errors = []
            for i in range(len(pred_dx)):
                # Obtener targets reales con padding
                target_dx_full, mask_dx = prepare_targets([y_va_dx[i]], cfg.MAX_FUTURE_HORIZON)
                target_dy_full, mask_dy = prepare_targets([y_va_dy[i]], cfg.MAX_FUTURE_HORIZON)
                
                target_dx_arr = target_dx_full[0].cpu().numpy()
                target_dy_arr = target_dy_full[0].cpu().numpy()
                mask_arr = mask_dx[0].cpu().numpy()
                
                # Solo calcular error en posiciones válidas (mask == 1)
                valid_indices = mask_arr > 0
                if valid_indices.sum() > 0:
                    dx_error = (pred_dx[i][valid_indices] - target_dx_arr[valid_indices]) ** 2
                    dy_error = (pred_dy[i][valid_indices] - target_dy_arr[valid_indices]) ** 2
                    squared_errors.extend(dx_error + dy_error)
            
            fold_rmse = np.sqrt(np.mean(squared_errors) / 2)
            fold_rmse_list.append(fold_rmse)
            
            print(f"Fold {fold} (semilla {seed}) — val loss: dx={loss_x:.5f}, dy={loss_y:.5f} | RMSE={fold_rmse:.5f}")

    # Calcular estadísticas de RMSE
    rmse_mean = np.mean(fold_rmse_list)
    rmse_std = np.std(fold_rmse_list)
    
    print("\n" + "="*80)
    print("ESTADÍSTICAS DE RMSE POR FOLD")
    print("="*80)
    for i, rmse in enumerate(fold_rmse_list, 1):
        print(f"Fold {i}: RMSE = {rmse:.5f}")
    print("-"*80)
    print(f"RMSE Promedio: {rmse_mean:.5f}")
    print(f"RMSE Desviación Estándar: {rmse_std:.5f}")
    print("="*80)

    # [4/4] Inferencia sobre el test usando todos los modelos entrenados
    print(f"\n[4/4] Inferencia y submission con ensemble de {len(all_models_x)} modelos...")
    test_seqs, test_meta, feat_cols_t, dir_map_test = prepare_sequences_with_advanced_features(
        test_input, test_template=test_template, is_training=False,
        window_size=cfg.WINDOW_SIZE
    )
    assert feat_cols_t == feat_cols, "¡Las columnas de características de Train/Test no coinciden!"

    idx_x = feat_cols.index('x')
    idx_y = feat_cols.index('y')

    X_test_raw = list(test_seqs)
    x_last_uni = np.array([s[-1, idx_x] for s in X_test_raw], dtype=np.float32)
    y_last_uni = np.array([s[-1, idx_y] for s in X_test_raw], dtype=np.float32)

    # Ensemble a través de todos los modelos de todos los folds y todas las semillas
    all_preds_dx, all_preds_dy = [], []
    
    # Iteramos sobre todos los modelos y escaladores guardados
    for mx, my, sc in zip(all_models_x, all_models_y, all_scalers):
        X_sc = np.stack([sc.transform(s) for s in X_test_raw]).astype(np.float32)
        X_t = torch.tensor(X_sc).to(cfg.DEVICE)
        mx.eval()
        my.eval()
        with torch.no_grad():
            all_preds_dx.append(mx(X_t).cpu().numpy())
            all_preds_dy.append(my(X_t).cpu().numpy())

    # Promediamos todas las predicciones
    ens_dx = np.mean(all_preds_dx, axis=0)
    ens_dy = np.mean(all_preds_dy, axis=0)
    H = ens_dx.shape[1]

    # Construcción de las filas para la submission, con inversión para jugadas a la derecha
    rows = []
    tt_idx = test_template.set_index(['game_id','play_id','nfl_id']).sort_index()

    for i, meta in enumerate(test_meta):
        gid = meta['game_id']; pid = meta['play_id']; nid = meta['nfl_id']
        play_dir = meta['play_direction']
        play_is_right = (play_dir == 'right')

        try:
            fids = tt_idx.loc[(gid,pid,nid),'frame_id']
            if isinstance(fids, pd.Series):
                fids = fids.sort_values().tolist()
            else:
                fids = [int(fids)]
        except KeyError:
            continue

        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            x_uni = np.clip(x_last_uni[i] + ens_dx[i, tt], 0, FIELD_LENGTH)
            y_uni = np.clip(y_last_uni[i] + ens_dy[i, tt], 0, FIELD_WIDTH)
            x_out, y_out = invert_to_original_direction(x_uni, y_uni, play_is_right)

            rows.append({
                'id': f"{gid}_{pid}_{nid}_{int(fid)}",
                'x': x_out,
                'y': y_out
            })

    submission = pd.DataFrame(rows)
    submission.to_csv("submission.csv", index=False)
    print("\n" + "="*80)
    print("¡PASO 2 COMPLETO!")
    print("="*80)
    print(f"✓ Submission guardada en submission.csv  |  Filas: {len(submission)}")
    print(f"Total de modelos en ensemble: {len(all_models_x)}")
    print(f"Características utilizadas: {len(feat_cols)}  (cuDF activo: {USE_CUDF})")

if __name__ == "__main__":
    main()
