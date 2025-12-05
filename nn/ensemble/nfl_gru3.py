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
from shutil import make_archive
import json
import random
import joblib
from glob import glob
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

    SAVE_DIR = Path("./saved_models")       
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    
    MODELS_DIR = Path("/kaggle/input/hsiaosuan-sttn/saved_models")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    TRAIN = False
    SUB   = True

    #是否使用Bi-GRU
    BIDIRECTIONAL = True

    #显式指定特征组（确保 train/sub 一致）
    FEATURE_GROUPS = [
        'distance_rate','target_alignment','multi_window_rolling','extended_lags',
        'velocity_changes','field_position','role_specific','time_features','jerk_features','interaction_features_mid','qb_relative',
        # 'interaction_features', #保守版本对抗特征
        # 'curvature_land_features',  #若线上无落点，勿开启
    ]

    #Training Setting
    SEED = 42
    SEEDS = [42, 19, 89, 64]   #多种子集成
    N_FOLDS = 5
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3

    WINDOW_SIZE = 10
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94  #不要动这个东西

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(Config.SEED)

def compute_val_rmse(mx, my, X_val_sc, ydx_list, ydy_list, horizon, device, mode="per-dim"):
    """
    计算验证集轨迹误差。
    mode:
      - "per-dim"  : sqrt( ( (dx^2+dy^2) 总和 / (2N) ) )  ← 你截图里的公式（推荐用于对齐）
      - "2d"       : sqrt( ( (dx^2+dy^2) 总和 / N ) )     ← 二维欧氏 RMSE（会比 per-dim 大 sqrt(2)）
      - "mean-dist": 平均欧氏距离 E[ sqrt(dx^2+dy^2) ]     ← 某些比赛用这个口径
    N = 有效的样本×时间步（按 mask 统计）
    """
    X_t = torch.tensor(X_val_sc.astype(np.float32)).to(device)
    with torch.no_grad():
        pdx = mx(X_t).cpu().numpy()   # (N, H)
        pdy = my(X_t).cpu().numpy()   # (N, H)

    ydx, m = prepare_targets(ydx_list, horizon)  # (N,H), (N,H)
    ydy, _ = prepare_targets(ydy_list, horizon)
    ydx, ydy, m = ydx.numpy(), ydy.numpy(), m.numpy()

    se_sum2d = ((pdx - ydx)**2 + (pdy - ydy)**2) * m
    denom = m.sum() + 1e-8

    if mode == "per-dim":
        return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))
    elif mode == "2d":
        return float(np.sqrt(se_sum2d.sum() / denom))
    elif mode == "mean-dist":
        dist = np.sqrt(se_sum2d)  # 元素级开根号
        return float(dist.sum() / denom)
    else:
        raise ValueError("mode must be one of {'per-dim','2d','mean-dist'}")

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
            'interaction_features': self._create_interaction_features, #添加,交互对抗特征
            'interaction_features_mid': self._create_interaction_features_mid,  #激进的交互对抗特征
            'qb_relative': self._create_qb_relative_features, #相对于四分卫位置的建模
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
        df['time_sin'] = np.sin(2*np.pi*df['normalized_time'])
        df['time_cos'] = np.cos(2*np.pi*df['normalized_time'])
        return df, ['frames_elapsed','normalized_time','time_sin','time_cos']

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
        """
        -落点侧向偏差（符号）：landing_point 相对“当前运动方向”的左右偏离
          lateral = cross(u_dir, vector_to_land)（>0 表示落点在运动方向左侧）
        -bearing_to_land_signed: 运动方向 vs 落点方位角
        -速度归一化曲率： wrap(Δdir)/ (s*Δt) ，窗口化(3/5) 的均值/绝对值
        """
        import numpy as np
        # 侧向偏差 & bearing_to_land
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

    def _create_interaction_features(self, df, speed_eps=0.5):
        """
        Lite Receiver–Defender interaction features (conservative):
          - Only K=1 (nearest opponent)
          - Compute ONLY for target player (player_to_predict==True) if column exists
          - Features:
              opp_dmin        : distance to nearest opposite-side opponent (clipped [0,30])
              opp_close_rate  : (v_opp - v_self) projected onto (opp->self) unit vector (clipped [-10,10])
              opp_leverage    : sign of cross( self_vel , self->opp ) in 2D ( {-1,0,1} ), gated by speed
        """
        import numpy as np
    
        need = ['x','y','velocity_x','velocity_y','player_side','frame_id']
        for c in need:
            if c not in df.columns:
                return df, []  # missing columns → skip safely
    
        out_cols = ['opp_dmin','opp_close_rate','opp_leverage']
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        key = ['game_id','play_id','frame_id']
        use_mask_global = ('player_to_predict' in df.columns)
    
        for _, g in df.groupby(key, sort=False):
            idx = g.index.values
            if len(g) <= 1:
                continue
    
            pos = g[['x','y']].values.astype('float32')          # (N,2)
            vel = g[['velocity_x','velocity_y']].values.astype('float32')
            side_off = (g['player_side'].values == 'Offense')    # (N,)
            side_def = ~side_off
    
            # only compute for target players if available
            if use_mask_global:
                tgt_mask = g['player_to_predict'].astype(bool).values
            else:
                tgt_mask = np.ones(len(g), dtype=bool)
    
            def _assign(A_mask, B_mask):
                A_mask = A_mask & tgt_mask
                A_idx = np.where(A_mask)[0]
                B_idx = np.where(B_mask)[0]
                if len(A_idx)==0 or len(B_idx)==0:
                    return
    
                Apos, Bpos = pos[A_idx], pos[B_idx]
                Avel, Bvel = vel[A_idx], vel[B_idx]
    
                # pairwise distances (K=1)
                dx = Apos[:,None,0] - Bpos[None,:,0]
                dy = Apos[:,None,1] - Bpos[None,:,1]
                D  = np.sqrt(dx*dx + dy*dy) + 1e-6                 # (Na,Nb)
                j  = np.argmin(D, axis=1)                          # nearest opponent
    
                dmin = D[np.arange(len(A_idx)), j]
                dmin = np.clip(dmin, 0.0, 30.0)                    # robust clip
    
                # closing rate
                r   = Apos - Bpos[j]                                # opp->self
                u   = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-6)
                v_rel = Bvel[j] - Avel
                close = np.einsum('ij,ij->i', v_rel, u)
                close = np.clip(close, -10.0, 10.0)
    
                # leverage sign, gated by own speed
                speed   = np.linalg.norm(Avel, axis=1)
                to_opp  = Bpos[j] - Apos                            # self->opp
                cross_z = to_opp[:,0]*Avel[:,1] - to_opp[:,1]*Avel[:,0]
                lever   = np.where(speed > speed_eps, np.sign(cross_z), 0).astype('int8')
    
                rows = idx[A_idx]
                df.loc[rows, 'opp_dmin']       = dmin
                df.loc[rows, 'opp_close_rate'] = close
                df.loc[rows, 'opp_leverage']   = lever
    
            # Offense w.r.t. Defense
            _assign(side_off, side_def)
            # Defense w.r.t. Offense
            _assign(side_def, side_off)
    
        return df, out_cols

    def _create_interaction_features_mid(self, df, speed_eps=0.5, k=2, radius=10.0):
        """
        Medium-aggressive Receiver–Defender interaction:
          - K=2 nearest opposite-side opponents (still per-frame)
          - Keep target-only computation if 'player_to_predict' exists
          - New features (small set):
              opp_d2                 : 2nd nearest distance (clipped [0,30])
              opp_dmean_k2           : mean distance of top-2
              opp_close_rate_min_k2  : min closing rate among top-2 (more threatening)
              opp_pursuit_error_min_k2: min |angle( v_opp , dir to self )| in degrees
              opp_density_r10        : #opponents within radius (10 yards)
          - Light temporal smoothing on previous base features:
              opp_dmin_roll3, opp_close_rate_roll3
        """
    
        need = ['x','y','velocity_x','velocity_y','player_side','frame_id']
        for c in need:
            if c not in df.columns:
                return df, []  # safe skip
    
        # 保证保守版的三列存在（以便 rolling）
        base_cols = ['opp_dmin','opp_close_rate','opp_leverage']
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        # 新增列（初始化）
        new_cols = ['opp_d2','opp_dmean_k2','opp_close_rate_min_k2',
                    'opp_pursuit_error_min_k2','opp_density_r10','ally_density_r10']
        for c in new_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        key = ['game_id','play_id','frame_id']
        use_mask_global = ('player_to_predict' in df.columns)
    
        def angle_between(v1, v2, eps=1e-6):
            # 返回弧度的夹角 (0..pi)
            dot = np.einsum('ij,ij->i', v1, v2)
            n1  = np.linalg.norm(v1, axis=1) + eps
            n2  = np.linalg.norm(v2, axis=1) + eps
            cos = np.clip(dot / (n1*n2), -1.0, 1.0)
            return np.arccos(cos)
    
        for _, g in df.groupby(key, sort=False):
            idx = g.index.values
            if len(g) <= 1:
                continue
    
            pos = g[['x','y']].values.astype('float32')
            vel = g[['velocity_x','velocity_y']].values.astype('float32')
            side_off = (g['player_side'].values == 'Offense')
            side_def = ~side_off
    
            # target-only
            if use_mask_global:
                tgt_mask = g['player_to_predict'].astype(bool).values
            else:
                tgt_mask = np.ones(len(g), dtype=bool)
    
            def _assign(A_mask, B_mask):
                A_mask = A_mask & tgt_mask
                A_idx = np.where(A_mask)[0]
                B_idx = np.where(B_mask)[0]
                if len(A_idx)==0 or len(B_idx)==0:
                    return
    
                Apos, Bpos = pos[A_idx], pos[B_idx]
                Avel, Bvel = vel[A_idx], vel[B_idx]
    
                # pairwise distances
                dx = Apos[:,None,0] - Bpos[None,:,0]
                dy = Apos[:,None,1] - Bpos[None,:,1]
                D  = np.sqrt(dx*dx + dy*dy) + 1e-6          # (Na,Nb)
                # Top-2 indices along last axis
                k_use = min(k, D.shape[1])
                part = np.argpartition(D, kth=range(k_use), axis=1)[:, :k_use]  # (Na, k_use)
                # Gather top-2 distances and opponent indices
                rows = np.arange(len(A_idx))[:,None]
                d_top = D[rows, part]                 # (Na, k_use)
                jidx  = part                          # opponent indices for each A
    
                # 基于 K=2 推导的派生量
                d_sorted = np.sort(d_top, axis=1)     # 升序：d1, d2
                d1 = d_sorted[:,0]
                d2 = d_sorted[:,1] if k_use >= 2 else d_sorted[:,0]
    
                # 关闭率：对每个所选对手计算，再取 min
                # opp->self 单位向量
                # 先取最近对手用于 opp_dmin/close_rate（与保守版保持一致含义）
                j1 = jidx[np.arange(len(A_idx)), np.argmin(d_top, axis=1)]
                r1 = Apos - Bpos[j1]                         # opp->self
                u1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-6)
                vrel1 = Bvel[j1] - Avel
                close1 = np.einsum('ij,ij->i', vrel1, u1)
                close1 = np.clip(close1, -10.0, 10.0)
    
                #第二近对手的关闭率
                if k_use >= 2:
                    # 取第二近（不是最小值位置）
                    order = np.argsort(d_top, axis=1)
                    j2 = jidx[rows[:,0], order[:,1]]
                    r2 = Apos - Bpos[j2]
                    u2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-6)
                    vrel2 = Bvel[j2] - Avel
                    close2 = np.einsum('ij,ij->i', vrel2, u2)
                    close2 = np.clip(close2, -10.0, 10.0)
                    close_min_k2 = np.minimum(close1, close2)
                else:
                    close2 = close1
                    close_min_k2 = close1
    
                #pursuit error：对手速度 vs 指向我方连线（取最小绝对值，单位度）
                # 角度=atan2(|cross|, dot)≡arccos(dot/(||v||·||u||))
                ang1 = angle_between(Bvel[j1], u1) * 180.0/np.pi
                if k_use >= 2:
                    ang2 = angle_between(Bvel[j2], u2) * 180.0/np.pi
                    perr_min_k2 = np.minimum(np.abs(ang1), np.abs(ang2))
                else:
                    perr_min_k2 = np.abs(ang1)
    
                #局部密度（半径内的对手数）
                density = (D <= radius).sum(axis=1).astype('float32')
                #同侧密度（A方内的 self-self 距离）；简单起见：用 Apos 两两距离
                rows_idx = idx[A_idx]  
                
                # 同侧密度
                try:
                    from scipy.spatial.distance import cdist
                    A2A = cdist(Apos, Apos) + 1e-6
                except Exception:
                    # 无 scipy 的 fallback
                    diff = Apos[:,None,:] - Apos[None,:,:]
                    A2A = np.sqrt((diff * diff).sum(-1)) + 1e-6
                same_density = (A2A <= radius).sum(axis=1) - 1  # 自身-1
                
                # 写回
                df.loc[rows_idx, 'ally_density_r10'] = same_density.astype('float32')
                df.loc[rows_idx, 'opp_dmin']       = np.clip(d1, 0.0, 30.0)
                df.loc[rows_idx, 'opp_close_rate'] = close1
                df.loc[rows_idx, 'opp_d2']                 = np.clip(d2, 0.0, 30.0)
                df.loc[rows_idx, 'opp_dmean_k2']           = (np.clip(d1,0,30)+np.clip(d2,0,30))/2.0
                df.loc[rows_idx, 'opp_close_rate_min_k2']  = close_min_k2
                df.loc[rows_idx, 'opp_pursuit_error_min_k2']= perr_min_k2
                df.loc[rows_idx, 'opp_density_r10']        = density
    
            # Offense vs Defense
            _assign(side_off, side_def)
            # Defense vs Offense
            _assign(side_def, side_off)
    
        # 轻量时序平滑（3 帧 rolling）
        # 注意：以个体维度分组，不跨人
        for col in ['opp_dmin','opp_close_rate']:
            if col in df.columns:
                r = (
                    df.groupby(self.gcols)[col]
                      .rolling(3, min_periods=1).mean()
                      .reset_index(level=list(range(len(self.gcols))), drop=True)
                )
                df[f'{col}_roll3'] = r
    
        out_cols = base_cols + new_cols + ['opp_dmin_roll3','opp_close_rate_roll3']
        # 只返回本函数“新增/更新”的列（如果有的来自保守版就不重复计数也没关系）
        out_cols = [c for c in out_cols if c in df.columns]
        return df, out_cols

    def _create_qb_relative_features(self, df):
        """
        QB-relative geometry (per frame):
          - qb_distance
          - vel_to_qb_alignment, vel_to_qb_perp
          - bearing_to_qb_signed (player facing vs vector to QB)
        仅依赖: x,y,velocity_x,velocity_y,dir,player_role,frame_id
        """
        need = ['x','y','velocity_x','velocity_y','dir','player_role','frame_id']
        for c in need:
            if c not in df.columns:
                return df, []  # 缺列则安全跳过
    
        out_cols = ['qb_distance','vel_to_qb_alignment','vel_to_qb_perp','bearing_to_qb_signed','bearing_to_qb_sin','bearing_to_qb_cos']
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan
    
        key = ['game_id','play_id','frame_id']
        for _, g in df.groupby(key, sort=False):
            idx = g.index.values
    
            # 找本帧 QB（通常唯一；若多于1取第一个；找不到则跳过）
            qb_rows = g[g['player_role'] == 'Passer']
            if qb_rows.empty:
                continue
            qb_x = float(qb_rows.iloc[0]['x'])
            qb_y = float(qb_rows.iloc[0]['y'])
    
            dx = g['x'].values.astype('float32') - qb_x
            dy = g['y'].values.astype('float32') - qb_y
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            ux, uy = dx/dist, dy/dist  # QB->player 单位向量
    
            vx = g['velocity_x'].values.astype('float32')
            vy = g['velocity_y'].values.astype('float32')
    
            align = vx*ux + vy*uy
            perp  = vx*(-uy) + vy*ux
    
            # bearing 差：玩家朝向 vs 指向 QB 的方向（有符号，(-180,180]）
            dir_rad = np.deg2rad(g['dir'].fillna(0.0).astype('float32').values)
            to_qb_angle = np.arctan2(-dy, -dx)  # player->QB
            bearing = np.rad2deg(np.arctan2(np.sin(to_qb_angle - dir_rad),
                                            np.cos(to_qb_angle - dir_rad)))
    
            df.loc[idx, 'qb_distance'] = dist
            df.loc[idx, 'vel_to_qb_alignment'] = align
            df.loc[idx, 'vel_to_qb_perp'] = perp
            df.loc[idx, 'bearing_to_qb_signed'] = bearing
            df.loc[idx, 'bearing_to_qb_sin'] = np.sin(np.deg2rad(bearing))
            df.loc[idx, 'bearing_to_qb_cos'] = np.cos(np.deg2rad(bearing))
    
        return df, out_cols

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


#[A] 统一键类型的辅助函数
def _canonicalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ('game_id','play_id','nfl_id'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 丢掉缺失键
    df = df.dropna(subset=['game_id','play_id','nfl_id'])
    # 统一为 int64
    df['game_id'] = df['game_id'].astype('int64')
    df['play_id'] = df['play_id'].astype('int64')
    df['nfl_id']  = df['nfl_id'].astype('int64')
    return df

def prepare_sequences_with_advanced_features(
        input_df, output_df=None, test_template=None, 
        is_training=True, window_size=10, feature_groups=None):

    print(f"\n{'='*80}")
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES (UNIFIED FRAME)")
    print(f"{'='*80}")
    print(f"Window size: {window_size}")

    # --- [B] 先统一键类型（输入/输出/测试模板都处理） ---
    input_df  = _canonicalize_key_dtypes(input_df)
    if is_training:
        assert output_df is not None
        output_df = _canonicalize_key_dtypes(output_df)
    else:
        assert test_template is not None
        test_template = _canonicalize_key_dtypes(test_template)

    if feature_groups is None:
        feature_groups = [
            'distance_rate','target_alignment','multi_window_rolling','extended_lags',
            'velocity_changes','field_position','role_specific','time_features',
            'jerk_features','interaction_features_mid'
        ]

    # --- Direction map & unify ---
    dir_map   = build_play_direction_map(input_df)
    input_df_u= unify_left_direction(input_df)

    if is_training:
        out_u = apply_direction_to_df(output_df, dir_map)
        target_rows   = out_u
        target_groups = out_u[['game_id','play_id','nfl_id']].drop_duplicates()
    else:
        if 'play_direction' not in test_template.columns:
            dir_df = dir_map.reset_index()
            test_template = test_template.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
        target_rows   = test_template
        target_groups = test_template[['game_id','play_id','nfl_id','play_direction']].drop_duplicates()

    assert target_rows[['game_id','play_id','play_direction']].isna().sum().sum() == 0, \
        "play_direction merge failed; check (game_id, play_id) coverage"
    print("play_direction merge OK:", target_rows['play_direction'].value_counts(dropna=False).to_dict())

    # --- FE ---
    fe = FeatureEngineer(feature_groups)
    processed_df, feature_cols = fe.transform(input_df_u)

    # --- Build sequences ---
    print("\nStep 3/3: Creating sequences...")
    processed_df = processed_df.set_index(['game_id','play_id','nfl_id']).sort_index()
    grouped = processed_df.groupby(level=['game_id','play_id','nfl_id'])

    # [C] 可选：打印键覆盖率，快速定位 miss 的真正原因
    avail_keys = (
        processed_df.reset_index()[['game_id','play_id','nfl_id']]
        .drop_duplicates()
    )
    inter = target_groups[['game_id','play_id','nfl_id']].merge(
        avail_keys, on=['game_id','play_id','nfl_id'], how='inner'
    )
    print(f"[COVERAGE] target_keys={len(target_groups)} | "
          f"input_keys={len(avail_keys)} | "
          f"matched={len(inter)}")

    # helpful indices
    idx_x = feature_cols.index('x')
    idx_y = feature_cols.index('y')

    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []

    it = target_groups.itertuples(index=False)
    it = tqdm(list(it), total=len(target_groups), desc="Creating sequences")

    for row in it:
        gid = row[0]; pid = row[1]; nid = row[2]
        play_dir = row[3] if (not is_training and len(row) >= 4) else None
        key = (gid, pid, nid)

        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue

        input_window = group_df.tail(window_size)

        # --- [D] 训练端也允许左侧填充（与测试一致），避免全被 <window_size 丢弃 ---
        if len(input_window) < window_size:
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        # input_window = input_window.fillna(group_df.mean(numeric_only=True))
        input_window = input_window.fillna(input_window.mean(numeric_only=True))
        seq = input_window[feature_cols].values

        if np.isnan(seq).any():
            seq = np.nan_to_num(seq, nan=0.0)

        sequences.append(seq)

        if is_training:
            out_grp = target_rows[
                (target_rows['game_id']==gid) &
                (target_rows['play_id']==pid) &
                (target_rows['nfl_id']==nid)
            ].sort_values('frame_id')
            if len(out_grp)==0:
                sequences.pop()  # 回滚
                continue

            last_x = seq[-1, idx_x]
            last_y = seq[-1, idx_y]
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y

            targets_dx.append(dx.astype(np.float32))
            targets_dy.append(dy.astype(np.float32))
            targets_fids.append(out_grp['frame_id'].values.astype(np.int32))

        seq_meta.append({
            'game_id': gid,
            'play_id': pid,
            'nfl_id': nid,
            'frame_id': int(input_window.iloc[-1]['frame_id']) if len(input_window) else -1,
            'play_direction': (None if is_training else play_dir),
        })

    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")

    if is_training:
        return sequences, targets_dx, targets_dy, targets_fids, seq_meta, feature_cols, dir_map
    return sequences, seq_meta, feature_cols, dir_map

# -------------------------------
# Tools for model saving & loading
# -------------------------------
def _seed_dir(base_dir: Path, seed: int) -> Path:
    d = base_dir / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_fold_artifacts(seed:int, fold:int, scaler, mx:nn.Module, my:nn.Module, base_dir:Path):
    sdir = _seed_dir(base_dir, seed)
    joblib.dump(scaler, sdir / f"scaler_fold{fold}.pkl")
    torch.save(mx.state_dict(), sdir / f"model_dx_fold{fold}.pt")
    torch.save(my.state_dict(), sdir / f"model_dy_fold{fold}.pt")

def write_meta(feature_cols:list, cfg:Config, base_dir:Path):
    meta = {
        "seeds": cfg.SEEDS,
        "n_folds": cfg.N_FOLDS,
        "feature_cols": feature_cols,
        "window_size": cfg.WINDOW_SIZE,
        "max_future_horizon": cfg.MAX_FUTURE_HORIZON,
        "feature_groups": cfg.FEATURE_GROUPS,
        "version": 1,
        "hidden_dim": cfg.HIDDEN_DIM,
        "bidirectional": getattr(cfg, "BIDIRECTIONAL", False),
    }
    with open(base_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[META] wrote meta.json to {base_dir}")

def load_saved_ensemble(cfg:Config, base_dir:Path):
    meta_path = base_dir / "meta.json"
    assert meta_path.exists(), f"meta.json not found: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    horizon = int(meta["max_future_horizon"])
    seeds   = meta["seeds"]
    n_folds = int(meta["n_folds"])
    hidden_dim   = int(meta.get("hidden_dim", 128))
    bidirectional= bool(meta.get("bidirectional", False))

    models_x, models_y, scalers = [], [], []
    for seed in seeds:
        sdir = base_dir / f"seed_{seed}"
        for fold in range(1, n_folds + 1):
            sc_path = sdir / f"scaler_fold{fold}.pkl"
            dx_path = sdir / f"model_dx_fold{fold}.pt"
            dy_path = sdir / f"model_dy_fold{fold}.pt"
            if not (sc_path.exists() and dx_path.exists() and dy_path.exists()):
                print(f"[WARN] missing seed={seed} fold={fold}, skip")
                continue
            scaler = joblib.load(sc_path)
            mx = SeqModel(len(feature_cols), horizon,
                          hidden_dim=hidden_dim, bidirectional=bidirectional
                          ).to(cfg.DEVICE)
            mx.load_state_dict(torch.load(dx_path, map_location=cfg.DEVICE)); mx.eval()
            my = SeqModel(len(feature_cols), horizon,
                          hidden_dim=hidden_dim, bidirectional=bidirectional
                          ).to(cfg.DEVICE)
            my.load_state_dict(torch.load(dy_path, map_location=cfg.DEVICE)); my.eval()
            scalers.append(scaler); models_x.append(mx); models_y.append(my)

    assert len(models_x) > 0, f"No models loaded from {base_dir}"
    print(f"[LOAD] loaded {len(models_x)} ΔX & {len(models_y)} ΔY models from {base_dir}")
    return models_x, models_y, scalers, meta

# -------------------------------
# Loss (Huber + time decay + 2nd-order velocity smooth)
# -------------------------------
class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.03, lam_smooth=0.01):
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
            self.delta * (abs_err - 0.5 * self.delta)
        )

        # time decay (keep your logic)
        if self.time_decay and self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device, dtype=pred.dtype)
            w = torch.exp(-self.time_decay * t).view(1, L)
            huber = huber * w
            mask  = mask  * w

        main_loss = (huber * mask).sum() / (mask.sum() + 1e-8)

        # velocity smooth (2nd difference ≈ jerk), conservative mask对齐
        if self.lam_smooth and pred.size(1) > 2:
            d1 = pred[:, 1:] - pred[:, :-1]          # [B, T-1]
            d2 = d1[:, 1:] - d1[:, :-1]              # [B, T-2]
            m2 = mask[:, 2:]                         # 对齐长度
            smooth = (d2 * d2) * m2
            smooth_loss = smooth.sum() / (m2.sum() + 1e-8)
        else:
            smooth_loss = pred.new_tensor(0.0)

        return main_loss + self.lam_smooth * smooth_loss


# class SeqModel(nn.Module):
#     def __init__(self, input_dim, horizon):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
#         self.pool_ln = nn.LayerNorm(128)
#         self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
#         self.pool_query = nn.Parameter(torch.randn(1, 1, 128))
#         self.head = nn.Sequential(
#             nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, horizon)
#         )
#     def forward(self, x):
#         h, _ = self.gru(x)
#         B = h.size(0)
#         q = self.pool_query.expand(B, -1, -1)
#         ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
#         out = self.head(ctx.squeeze(1))
#         return torch.cumsum(out, dim=1)

class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_hidden, horizon, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.proj = nn.Linear(d_in, d_hidden)  # skip
        self.out = nn.Linear(d_hidden, horizon)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        y = self.drop(self.act(self.fc1(x)))
        y = self.drop(self.act(self.fc2(y)) + self.proj(x))  # 残差
        return self.out(y)

class SeqModel(nn.Module):
    def __init__(self, input_dim, horizon, hidden_dim=128, num_layers=2, bidirectional=False,
                 use_residual=True, n_queries=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=0.1, bidirectional=bidirectional
        )
        h_out = hidden_dim * (2 if bidirectional else 1)
        self.in_proj = nn.Linear(input_dim, h_out) if use_residual else None

        self.pool_ln   = nn.LayerNorm(h_out)
        self.pool_attn = nn.MultiheadAttention(h_out, num_heads=4, batch_first=True)
        self.pool_query= nn.Parameter(torch.randn(1, n_queries, h_out))  # 多query汇聚更多上下文
        self.head      = ResidualMLP(h_out*n_queries, hidden_dim, horizon)

    def forward(self, x):
        h, _ = self.gru(x)                        # [B,T,h_out]
        if self.use_residual:
            h = h + self.in_proj(x)               # 时间维残差

        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)     # [B,Q,h_out]
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))  # [B,Q,h_out]
        ctx = ctx.reshape(B, -1)                  # 拼成 [B, Q*h_out]
        out = self.head(ctx)                      # [B, H]
        return torch.cumsum(out, dim=1)


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

def train_model(X_train, y_train, X_val, y_val, input_dim, horizon, config, noise_std=0.01, model_kwargs=None):
    device = config.DEVICE
    # model = SeqModel(input_dim, horizon).to(device)
    model = SeqModel(input_dim, horizon, **(model_kwargs or {})).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

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
            # 训练期增强（与 TTA 对齐）
            bx = add_random_gaussian(bx, sigma_max=noise_std)  # 随机强度噪声
            bx = random_time_mask(bx, p=0.10, max_width=3)    # 时间mask
            bx = flip_context_keep_last(bx, p=0.10)           # 反转前T-1，末帧不动
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
            for bx, by, bm in va_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
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
def predict_with_tta_per_model(mx, my, scaler, X_test_raw, device, tta=6, noise_std=0.01, use_flip=True):
    """
    对单个 (mx,my,scaler) 做 TTA，返回 [N,H] 的 dx,dy 预测。
    - 在“标准化后的空间”加噪声（与训练一致）
    - 可选：反转前 T-1 帧（末帧不动），与未反转结果平均
    - 重复 tta 次取均值
    """
    mx.eval(); my.eval()
    outs_dx, outs_dy = [], []
    base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)

    for _ in range(max(1, tta)):
        xt_aug = xt
        if noise_std and noise_std > 0:
            xt_aug = xt_aug + torch.randn_like(xt_aug) * noise_std

        with torch.no_grad():
            dx = mx(xt_aug)
            dy = my(xt_aug)
            if use_flip and xt_aug.size(1) > 1:
                ctx = xt_aug[:, :-1].flip(1)
                xt_flip = torch.cat([ctx, xt_aug[:, -1:].clone()], dim=1)
                dx = 0.5 * (dx + mx(xt_flip))
                dy = 0.5 * (dy + my(xt_flip))

        outs_dx.append(dx.detach().cpu().numpy())
        outs_dy.append(dy.detach().cpu().numpy())

    return np.mean(outs_dx, axis=0), np.mean(outs_dy, axis=0)
import random as _py_random
import torch

def random_time_mask(bx, p=0.1, max_width=3):
    """
    在时间维做短段复制遮挡：随机挑一段 [s, s+w) 用前/后一帧值替换。
    - bx: [B, T, D] (torch, 支持在 GPU 上原地改)
    """
    if p <= 0 or max_width <= 0:
        return bx
    B, T, D = bx.shape
    if T <= 1:
        return bx
    for i in range(B):
        if _py_random.random() < p:
            w = _py_random.randint(1, max_width)
            s = _py_random.randint(0, max(0, T - 1 - w))
            if s > 0:
                bx[i, s:s+w] = bx[i, s-1].unsqueeze(0)
            else:
                bx[i, s:s+w] = bx[i, s+w].unsqueeze(0)
    return bx

def flip_context_keep_last(bx, p=0.1):
    """
    仅反转前 T-1 帧（保持最后一帧不动），制造“上下文反转”。
    """
    if p <= 0:
        return bx
    B, T, D = bx.shape
    if T <= 1:
        return bx
    mask = torch.rand(B, device=bx.device) < p
    if mask.any():
        ctx = bx[mask, :-1].flip(1)
        bx[mask] = torch.cat([ctx, bx[mask, -1:].clone()], dim=1)
    return bx

def add_random_gaussian(bx, sigma_max=0.02):
    """
    给整段序列加一次高斯噪声（强度在 [0, sigma_max] 内随机）。
    """
    if sigma_max <= 0:
        return bx
    sigma = sigma_max * torch.rand(1, device=bx.device)
    return bx + torch.randn_like(bx) * sigma
# ------------------------------_
# Main pipeline (MODIFICADO PARA ENSEMBLE DE SEMILLAS)
# ------------------------------_
class CFG(Config):
    # Añadimos la lista de semillas para el ensemble
    SEEDS = [42, 19, 89,64,33] # ¡Puedes cambiar o añadir más semillas aquí!

def main():
    cfg = CFG()
    print("="*80)
    print(f"RUN MODE: TRAIN={getattr(cfg, 'TRAIN', False)} | SUB={getattr(cfg, 'SUB', False)}")
    print("="*80)
    print(f"cuDF backend active? {USE_CUDF}")

    # --- sanity checks ---
    if not cfg.TRAIN and not cfg.SUB:
        raise ValueError("Please set a run mode: TRAIN=True/SUB=False or TRAIN=False/SUB=True")
    if cfg.TRAIN and cfg.SUB:
        raise ValueError("TRAIN and SUB cannot both be True")

    # ---------------------------
    # TRAIN: train & save to a writable dir
    # ---------------------------
    if cfg.TRAIN:
        # redirect saving path to a writable place
        save_dir = Path("./saved_models")
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg.MODELS_DIR = save_dir   # <<< key line: write into /kaggle/working

        # 1) load training data
        print("\n[1/4] 加载训练数据…")
        train_input_files  = [cfg.DATA_DIR / f"train/input_2023_w{w:02d}.csv"  for w in range(1, 19)]
        train_output_files = [cfg.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
        train_input  = pd.concat([pd.read_csv(f) for f in train_input_files  if f.exists()], ignore_index=True)
        train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()], ignore_index=True)

        # 2) features + sequences (unified direction)
        print("\n[2/4] 特征与序列（统一方向）…")
        feature_groups = getattr(cfg, "FEATURE_GROUPS", None)
        seqs, tdx, tdy, tfids, seq_meta, feat_cols, dir_map = prepare_sequences_with_advanced_features(
            train_input, output_df=train_output, is_training=True,
            window_size=cfg.WINDOW_SIZE, feature_groups=feature_groups
        )
        sequences  = list(seqs)
        targets_dx = list(tdx)
        targets_dy = list(tdy)

        # 2.5) write meta to the same (writable) dir
        write_meta(feat_cols, cfg, base_dir=cfg.MODELS_DIR)


        # 3) multi-seed × KFold, save per-fold artifacts
        print("\n[3/4] 多种子 × K 折训练并保存模型…")
        # groups = np.array([d['game_id'] for d in seq_meta])
        groups = np.array([f"{d['game_id']}_{d['play_id']}" for d in seq_meta])

        seeds = getattr(cfg, "SEEDS", [cfg.SEED])
        all_rmse = []        # 所有 seed×fold 的 per-dim RMSE
        cv_log = []          # 也把每折指标放进列表，最后写 json

        for seed in seeds:
            print(f"\n{'='*70}\n   Seed {seed}\n{'='*70}")
            set_seed(seed)
            gkf = GroupKFold(n_splits=cfg.N_FOLDS)

            fold_rmses = []  # 当前 seed 的每折 RMSE（per-dim）

            for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
                print(f"\n{'-'*60}\nFold {fold}/{cfg.N_FOLDS} (seed {seed})\n{'-'*60}")

                X_tr = [sequences[i] for i in tr]
                X_va = [sequences[i] for i in va]

                scaler = StandardScaler()
                scaler.fit(np.vstack([s for s in X_tr]))

                X_tr_sc = np.stack([scaler.transform(s) for s in X_tr]).astype(np.float32)
                X_va_sc = np.stack([scaler.transform(s) for s in X_va]).astype(np.float32)


                # model_kwargs = dict(hidden_dim=cfg.HIDDEN_DIM, bidirectional=getattr(cfg, "BIDIRECTIONAL", False))
                model_kwargs = dict(
                    hidden_dim=cfg.HIDDEN_DIM,
                    bidirectional=getattr(cfg, "BIDIRECTIONAL", False),
                    use_residual=True,
                    n_queries=2
                )
                
                print("训练 ΔX …")
                mx, loss_x = train_model(
                    X_tr_sc, [targets_dx[i] for i in tr],
                    X_va_sc, [targets_dx[i] for i in va],
                    X_tr_sc.shape[-1], cfg.MAX_FUTURE_HORIZON, cfg,
                    model_kwargs=model_kwargs
                )
                
                print("训练 ΔY …")
                my, loss_y = train_model(
                    X_tr_sc, [targets_dy[i] for i in tr],
                    X_va_sc, [targets_dy[i] for i in va],
                    X_tr_sc.shape[-1], cfg.MAX_FUTURE_HORIZON, cfg,
                    model_kwargs=model_kwargs
                )


                # --- NEW: 计算三口径，默认用 per-dim 进 CV 汇总 ---
                rmse_perdim = compute_val_rmse(
                    mx, my, X_va_sc,
                    [targets_dx[i] for i in va],
                    [targets_dy[i] for i in va],
                    cfg.MAX_FUTURE_HORIZON, cfg.DEVICE, mode="per-dim"
                )
                rmse_2d = compute_val_rmse(
                    mx, my, X_va_sc,
                    [targets_dx[i] for i in va],
                    [targets_dy[i] for i in va],
                    cfg.MAX_FUTURE_HORIZON, cfg.DEVICE, mode="2d"
                )
                mean_dist = compute_val_rmse(
                    mx, my, X_va_sc,
                    [targets_dx[i] for i in va],
                    [targets_dy[i] for i in va],
                    cfg.MAX_FUTURE_HORIZON, cfg.DEVICE, mode="mean-dist"
                )

                print(f"[VAL] seed {seed} fold {fold} → "
                      f"Huber dx={loss_x:.5f}, dy={loss_y:.5f} | "
                      f"per-dim RMSE={rmse_perdim:.4f} | 2D RMSE={rmse_2d:.4f} | meanDist={mean_dist:.4f} yards")

                fold_rmses.append(rmse_perdim)
                all_rmse.append(rmse_perdim)
                cv_log.append({
                    "seed": seed, "fold": fold,
                    "rmse_perdim": rmse_perdim,
                    "rmse_2d": rmse_2d,
                    "mean_dist": mean_dist,
                    "loss_dx": float(loss_x),
                    "loss_dy": float(loss_y),
                })

                # 保存模型
                save_fold_artifacts(seed=seed, fold=fold, scaler=scaler, mx=mx, my=my, base_dir=cfg.MODELS_DIR)

            # --- NEW: 当前 seed 汇总 ---
            print(f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
                  f"mean={float(np.mean(fold_rmses)):.4f} yards")

        # --- NEW: 所有 seeds×folds 的最终汇总 & 落盘 ---
        print(f"[CV SUMMARY] all folds RMSEs: {[f'{r:.4f}' for r in all_rmse]}")
        print(f"[CV SUMMARY] overall mean RMSE = {float(np.mean(all_rmse)):.4f} yards")

        # 写到磁盘（方便回看）
        try:
            with open(cfg.MODELS_DIR / "cv_metrics.json", "w") as f:
                json.dump({"per_fold": cv_log, "overall_mean_perdim": float(np.mean(all_rmse))}, f, indent=2)
            print(f"✓ CV metrics written to {cfg.MODELS_DIR / 'cv_metrics.json'}")
        except Exception as e:
            print(f"[WARN] writing cv_metrics.json failed: {e}")


        print("\n" + "="*80)
        print("COMPLETE (TRAIN)!")
        print("="*80)
        print(f"✓ Models saved to: {cfg.MODELS_DIR}")
        print(f"Seeds: {cfg.SEEDS} | Folds: {cfg.N_FOLDS} → checkpoints per axis: {len(cfg.SEEDS)*cfg.N_FOLDS}")
        print(f"Features used: {len(feat_cols)}  (cuDF active: {USE_CUDF})")
        return

    # ---------------------------
    # SUB: load from read-only input dir & infer
    # ---------------------------
    if cfg.SUB:
        # DO NOT change cfg.MODELS_DIR here — keep it as the dataset input path
        print("\n[1/3] 加载测试数据…")
        test_input    = pd.read_csv(cfg.DATA_DIR / "test_input.csv")
        test_template = pd.read_csv(cfg.DATA_DIR / "test.csv")

        print("\n[2/3] 读取已保存的模型与元信息…")
        # models_x, models_y, scalers, meta = load_saved_ensemble(cfg)
        models_x, models_y, scalers, meta = load_saved_ensemble(cfg, base_dir=cfg.MODELS_DIR)

        saved_feature_cols = meta["feature_cols"]
        saved_groups       = meta.get("feature_groups", getattr(cfg, "FEATURE_GROUPS", None))
        saved_window       = int(meta.get("window_size", cfg.WINDOW_SIZE))

        print("\n[3/3] 构建测试序列并推理（统一方向 → 反变换）…")
        test_seqs, test_meta, feat_cols_t, _ = prepare_sequences_with_advanced_features(
            test_input, test_template=test_template, is_training=False,
            window_size=saved_window, feature_groups=saved_groups
        )
        assert feat_cols_t == saved_feature_cols, \
            f"特征列不一致！训练: {len(saved_feature_cols)} vs 测试: {len(feat_cols_t)}"

        idx_x = feat_cols_t.index('x')
        idx_y = feat_cols_t.index('y')

        X_test_raw = list(test_seqs)
        x_last_uni = np.array([s[-1, idx_x] for s in X_test_raw], dtype=np.float32)
        y_last_uni = np.array([s[-1, idx_y] for s in X_test_raw], dtype=np.float32)

        # TTA across models
        tta_times   = 6           # 可调：4~8 都行
        tta_noise   = 0.01        # 与训练同量级或略小
        use_flip_ta = True
        
        all_preds_dx, all_preds_dy = [], []
        for mx, my, sc in zip(models_x, models_y, scalers):
            dx_tta, dy_tta = predict_with_tta_per_model(
                mx, my, sc, X_test_raw, cfg.DEVICE,
                tta=tta_times, noise_std=tta_noise, use_flip=use_flip_ta
            )
            all_preds_dx.append(dx_tta)
            all_preds_dy.append(dy_tta)
        
        ens_dx = np.mean(all_preds_dx, axis=0)
        ens_dy = np.mean(all_preds_dy, axis=0)

        H = ens_dx.shape[1]

        rows = []
        tt_idx = test_template.set_index(['game_id','play_id','nfl_id']).sort_index()
        for i, meta_row in enumerate(test_meta):
            gid = meta_row['game_id']; pid = meta_row['play_id']; nid = meta_row['nfl_id']
            play_is_right = (meta_row['play_direction'] == 'right')
            try:
                fids = tt_idx.loc[(gid,pid,nid),'frame_id']
                if isinstance(fids, pd.Series): fids = fids.sort_values().tolist()
                else: fids = [int(fids)]
            except KeyError:
                continue

            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                x_uni = np.clip(x_last_uni[i] + ens_dx[i, tt], 0, FIELD_LENGTH)
                y_uni = np.clip(y_last_uni[i] + ens_dy[i, tt], 0, FIELD_WIDTH)
                x_out, y_out = invert_to_original_direction(x_uni, y_uni, play_is_right)
                rows.append({'id': f"{gid}_{pid}_{nid}_{int(fid)}", 'x': x_out, 'y': y_out})

        submission = pd.DataFrame(rows)
        submission.to_csv("submission.csv", index=False)
        print("\n" + "="*80)
        print("COMPLETE (SUBMIT)!")
        print("="*80)
        print(f"✓ Submission saved to submission.csv  |  Rows: {len(submission)}")
        print(f"Total models in ensemble: {len(models_x)}")
        print(f"Features used: {len(saved_feature_cols)}  (cuDF active: {USE_CUDF})")
        return

    # 如果两个 flag 都没开，给出提醒
    raise ValueError("请在 Config 中设置运行模式：TRAIN=True/SUB=False 或 TRAIN=False/SUB=True")

if __name__ == "__main__":
    main()