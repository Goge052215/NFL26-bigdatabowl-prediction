import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

# Enhanced Config for Transformer
class TransformerConfig:
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("working")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    SEED = 42
    N_FOLDS = 5

    BATCH_SIZE = 128  # Reduced for Transformer memory requirements
    EPOCHS = 1000
    PATIENCE = 150
    LEARNING_RATE = 1e-4
    
    WINDOW_SIZE = 12  # Increased for better temporal context
    HIDDEN_DIM = 256
    MAX_FUTURE_HORIZON = 120
    
    # Transformer specific
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    FF_DIM = 512
    
    # GNN-lite parameters
    K_NEIGH = 5
    RADIUS = 15.0
    TAU = 6.0
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed=TransformerConfig.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(TransformerConfig.SEED)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, horizon, config=TransformerConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.horizon = horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.HIDDEN_DIM)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.HIDDEN_DIM, 
            max_len=config.WINDOW_SIZE
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.HIDDEN_DIM,
            nhead=config.N_HEADS,
            dim_feedforward=config.FF_DIM,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.N_LAYERS
        )
        
        # Enhanced spatial attention for better player interaction modeling
        self.spatial_attention = nn.MultiheadAttention(
            config.HIDDEN_DIM, 
            num_heads=8,  # Increased for better spatial modeling
            batch_first=True,
            dropout=config.DROPOUT
        )
        
        # Cross-attention for player interactions (when available)
        self.player_interaction_attention = nn.MultiheadAttention(
            config.HIDDEN_DIM,
            num_heads=4,
            batch_first=True,
            dropout=config.DROPOUT
        )
        
        # Multiple temporal queries for diverse temporal patterns
        self.num_temporal_queries = 4
        self.temporal_queries = nn.Parameter(torch.randn(self.num_temporal_queries, 1, config.HIDDEN_DIM))
        self.temporal_attention = nn.MultiheadAttention(
            config.HIDDEN_DIM,
            num_heads=config.N_HEADS,
            batch_first=True,
            dropout=config.DROPOUT
        )
        
        # Adaptive pooling for temporal queries
        self.query_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.layer_norm2 = nn.LayerNorm(config.HIDDEN_DIM)
        self.layer_norm3 = nn.LayerNorm(config.HIDDEN_DIM)
        
        # Enhanced prediction head with residual connections
        self.prediction_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT // 2),
            nn.Linear(config.HIDDEN_DIM // 2, horizon)
        )
        
        # Physics-informed velocity prediction
        self.velocity_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT // 2),
            nn.Linear(config.HIDDEN_DIM // 2, horizon)
        )
        
        # Learnable combination weights with better initialization
        self.position_weight = nn.Parameter(torch.tensor(0.6))
        self.velocity_weight = nn.Parameter(torch.tensor(0.4))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer encoding with residual connection
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        # Spatial attention (self-attention over sequence)
        spatial_out, _ = self.spatial_attention(encoded, encoded, encoded)
        spatial_out = self.layer_norm1(spatial_out + encoded)
        
        # Multiple temporal queries for diverse temporal patterns
        temporal_outputs = []
        for i in range(self.num_temporal_queries):
            query = self.temporal_queries[i].expand(batch_size, -1, -1)
            temporal_out, _ = self.temporal_attention(query, spatial_out, spatial_out)
            temporal_outputs.append(temporal_out)
        
        # Combine multiple temporal queries using adaptive pooling
        combined_temporal = torch.cat(temporal_outputs, dim=1)  # (batch_size, num_queries, hidden_dim)
        
        # Pool temporal queries to single representation
        pooled_temporal = self.query_pooling(combined_temporal.transpose(1, 2)).transpose(1, 2)
        pooled_temporal = pooled_temporal.squeeze(1)  # (batch_size, hidden_dim)
        
        # Apply final layer normalization
        pooled_temporal = self.layer_norm3(pooled_temporal)
        
        # Generate predictions
        position_pred = self.prediction_head(pooled_temporal)  # (batch_size, horizon)
        velocity_pred = self.velocity_head(pooled_temporal)    # (batch_size, horizon)
        
        # Physics-informed combination with proper weight normalization
        alpha = torch.sigmoid(self.position_weight)
        beta = torch.sigmoid(self.velocity_weight)
        
        # Ensure weights sum to 1
        total_weight = alpha + beta
        alpha = alpha / total_weight
        beta = beta / total_weight
        
        # Combine position and velocity predictions
        combined_pred = alpha * position_pred + beta * torch.cumsum(velocity_pred, dim=1)
        
        return combined_pred

class TemporalHuberLoss(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.02, velocity_weight=0.1):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.velocity_weight = velocity_weight
    
    def forward(self, pred, target, mask):
        # Standard Huber loss
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 
                           0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        # Temporal weighting (emphasize near-future predictions)
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L)
            huber = huber * weight
            mask = mask * weight
        
        # Velocity consistency loss (physics constraint)
        if self.velocity_weight > 0 and pred.size(1) > 1:
            velocity_pred = torch.diff(pred, dim=1)
            velocity_target = torch.diff(target, dim=1)
            velocity_loss = torch.abs(velocity_pred - velocity_target)
            
            # Add velocity loss to the main loss
            velocity_mask = mask[:, 1:]  # Adjust mask for diff operation
            velocity_component = (velocity_loss * velocity_mask).sum() / (velocity_mask.sum() + 1e-8)
            huber_component = (huber * mask).sum() / (mask.sum() + 1e-8)
            
            return huber_component + self.velocity_weight * velocity_component
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)

# Dataset class remains the same as in nn-mps.py
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

def collate_fn(batch):
    sequences, targets, masks = zip(*batch)
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    masks = torch.stack(masks)
    return sequences, targets, masks

def train_transformer_model(X_train, y_train, X_val, y_val, input_dim, horizon, config):
    device = config.DEVICE
    model = SpatioTemporalTransformer(input_dim, horizon, config).to(device)
    
    criterion = TemporalHuberLoss(delta=0.5, time_decay=0.02, velocity_weight=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler with warm restarts
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS//4, eta_min=1e-6)
    
    # Create datasets and dataloaders
    train_dataset = NFLDataset(X_train, y_train, horizon)
    val_dataset = NFLDataset(X_val, y_val, horizon)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    print(f"Training Transformer with {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(1, config.EPOCHS + 1):
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
            
            # Gradient clipping for stability
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
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
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

# Enhanced feature engineering functions
def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

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

        # Join neighbors at the ego's last_frame_id
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

        # Drop self
        tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]

        # Relative vectors
        tmp["dx"]  = tmp["x_nb"] - tmp["x"]
        tmp["dy"]  = tmp["y_nb"] - tmp["y"]
        tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
        tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
        tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)

        tmp = tmp[np.isfinite(tmp["dist"])]

        # Filter by radius and get top-K
        tmp = tmp[tmp["dist"] <= TransformerConfig.RADIUS]
        tmp = (tmp.sort_values(["game_id","play_id","nfl_id","dist"])
                  .groupby(["game_id","play_id","nfl_id"])
                  .head(TransformerConfig.K_NEIGH))

        # Compute embeddings with attention weights
        tmp["weight"] = np.exp(-tmp["dist"] / TransformerConfig.TAU)
        
        # Aggregate neighbor features
        agg_funcs = {
            "dx": ["mean", "std", "min", "max"],
            "dy": ["mean", "std", "min", "max"], 
            "dvx": ["mean", "std"],
            "dvy": ["mean", "std"],
            "dist": ["mean", "min"],
            "weight": ["sum"]
        }
        
        neighbor_agg = (tmp.groupby(["game_id","play_id","nfl_id"])
                           .agg(agg_funcs)
                           .reset_index())
        
        # Flatten column names
        neighbor_agg.columns = [
            "_".join(col).strip() if col[1] else col[0] 
            for col in neighbor_agg.columns.values
        ]
        
        # Merge back to original
        result = last.merge(neighbor_agg, on=["game_id","play_id","nfl_id"], how="left")
        
        # Fill missing values
        neighbor_cols = [c for c in result.columns if any(x in c for x in ["dx_", "dy_", "dvx_", "dvy_", "dist_", "weight_"])]
        result[neighbor_cols] = result[neighbor_cols].fillna(0)
        
        return result

def add_advanced_features(df):
    df = df.copy()
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    print("Adding advanced features...")
    
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
    for window in [3, 5, 8]:  # Adjusted for longer sequences
        for col in ['velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_roll{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_std{window}'] = df.groupby(gcols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                ).fillna(0)
    
    # GROUP 4: Extended Lag Features (12)
    for lag in [1, 2, 3, 4, 5, 6]:  # More lags for longer sequences
        for col in ['x', 'y']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(gcols)[col].shift(lag).fillna(0)
    
    # GROUP 5: Velocity and Acceleration Features (8)
    if 'velocity_x' in df.columns:
        df['velocity_x_change'] = df.groupby(gcols)['velocity_x'].diff().fillna(0)
        df['velocity_y_change'] = df.groupby(gcols)['velocity_y'].diff().fillna(0)
        df['speed_change'] = df.groupby(gcols)['s'].diff().fillna(0)
        df['direction_change'] = df.groupby(gcols)['dir'].diff().fillna(0)
        df['direction_change'] = df['direction_change'].apply(
            lambda x: x if abs(x) < 180 else x - 360 * np.sign(x)
        )
        
        # Acceleration features
        df['accel_magnitude'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
        df['jerk_x'] = df.groupby(gcols)['acceleration_x'].diff().fillna(0)
        df['jerk_y'] = df.groupby(gcols)['acceleration_y'].diff().fillna(0)
    
    # GROUP 6: Field Position Features (6)
    df['dist_from_left'] = df['y']
    df['dist_from_right'] = 53.3 - df['y']
    df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
    df['dist_from_endzone'] = np.minimum(df['x'], 120 - df['x'])
    df['field_zone_x'] = pd.cut(df['x'], bins=6, labels=False)  # Field zones
    df['field_zone_y'] = pd.cut(df['y'], bins=4, labels=False)
    
    # GROUP 7: Role-Specific Features (4)
    if 'is_receiver' in df.columns and 'velocity_alignment' in df.columns:
        df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
        df['receiver_deviation'] = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0))
    if 'is_coverage' in df.columns and 'closing_speed' in df.columns:
        df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']
        df['defender_angle_advantage'] = df['is_coverage'] * df.get('angle_to_ball', 0)
    
    # GROUP 8: Temporal Features (4)
    df['frames_elapsed'] = df.groupby(gcols).cumcount()
    df['normalized_time'] = df.groupby(gcols)['frames_elapsed'].transform(
        lambda x: x / (x.max() + 1)
    )
    df['time_since_snap'] = df['frames_elapsed'] * 0.1  # Convert to seconds
    df['play_progress'] = df['normalized_time']  # Alias for clarity
    
    print(f"Total features after enhancement: {len(df.columns)}")
    
    return df

def prepare_sequences_with_transformer_features(input_df, output_df=None, test_template=None, 
                                               is_training=True, window_size=TransformerConfig.WINDOW_SIZE):
    print(f"PREPARING SEQUENCES FOR TRANSFORMER")
    print(f"Window size: {window_size}")
    
    input_df = input_df.copy()
    
    # BASIC FEATURES
    print("Step 1/4: Adding basic features...")
    
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    
    # Enhanced physics calculations
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    delta_t = 0.1
    input_df['velocity_x'] = (input_df['s'] + 0.5 * input_df['a'] * delta_t) * np.sin(dir_rad)
    input_df['velocity_y'] = (input_df['s'] + 0.5 * input_df['a'] * delta_t) * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.sin(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.cos(dir_rad)
    
    # Player roles (enhanced)
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['is_rusher'] = (input_df['player_role'] == 'Pass Rush').astype(int)
    
    # Enhanced physics
    mass_kg = input_df['player_weight'].fillna(200.0) / 2.20462
    input_df['momentum_x'] = input_df['velocity_x'] * mass_kg
    input_df['momentum_y'] = input_df['velocity_y'] * mass_kg
    input_df['kinetic_energy'] = 0.5 * mass_kg * (input_df['s'] ** 2)
    input_df['power'] = input_df['kinetic_energy'] * input_df['a']  # Power = Energy * Acceleration
    
    # Ball features (enhanced)
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
        
        # Time to ball (physics-based)
        input_df['time_to_ball'] = input_df['distance_to_ball'] / (input_df['s'] + 0.1)
        input_df['ball_intercept_angle'] = np.abs(input_df['angle_to_ball'] - np.deg2rad(input_df['dir']))
    
    # Sort for temporal operations
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    # EMA features for smoother temporal modeling
    alpha = 0.3
    for col in ['velocity_x', 'velocity_y', 's', 'a']:
        if col in input_df.columns:
            input_df[f'{col}_ema'] = (input_df.groupby(gcols)[col]
                                     .transform(lambda x: x.ewm(alpha=alpha).mean()))
    
    # STEP 2: GNN Features
    print("Step 2/4: Adding GNN neighbor features...")
    gnn_processor = GNNLiteProcessor()
    neighbor_features = gnn_processor.compute_neighbor_embeddings(input_df)
    
    # Merge neighbor features
    input_df = input_df.merge(
        neighbor_features[['game_id', 'play_id', 'nfl_id'] + 
                        [c for c in neighbor_features.columns if 'dx_' in c or 'dy_' in c or 'dist_' in c]],
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    # STEP 3: Advanced Features
    print("Step 3/4: Adding advanced spatio-temporal features...")
    input_df = add_advanced_features(input_df)
    
    # STEP 4: Sequence Creation
    print("Step 4/4: Creating sequences...")
    
    # Select features for modeling (exclude identifiers and targets)
    exclude_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'player_name', 
                   'player_position', 'player_role', 'player_side', 'play_direction',
                   'player_birth_date', 'player_height', 'player_weight']
    
    # Encode play_direction as numeric before creating feature columns
    if 'play_direction' in input_df.columns:
        input_df['play_direction_encoded'] = (input_df['play_direction'] == 'right').astype(int)
    
    feature_cols = [c for c in input_df.columns if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for modeling")
    
    # Create sequences
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = [], [], [], [], []
    
    grouped = input_df.groupby(['game_id', 'play_id', 'nfl_id'])
    
    for (game_id, play_id, nfl_id), group in tqdm(grouped, desc="Creating sequences"):
        group = group.sort_values('frame_id').reset_index(drop=True)
        
        if len(group) < window_size:
            continue
            
        # Create sequence
        seq_data = group[feature_cols].values
        if len(seq_data) >= window_size:
            sequence = seq_data[-window_size:]  # Take last window_size frames
            sequences.append(sequence)
            
            sequence_ids.append({
                'game_id': game_id,
                'play_id': play_id, 
                'nfl_id': nfl_id
            })
            
            if is_training and output_df is not None:
                # Get targets
                target_data = output_df[
                    (output_df['game_id'] == game_id) &
                    (output_df['play_id'] == play_id) &
                    (output_df['nfl_id'] == nfl_id)
                ].sort_values('frame_id')
                
                if len(target_data) > 0:
                    # Get last position from input sequence
                    last_x = group.iloc[-1]['x']
                    last_y = group.iloc[-1]['y']
                    
                    # Compute dx, dy as differences from last position
                    dx_vals = target_data['x'].values - last_x
                    dy_vals = target_data['y'].values - last_y
                    frame_ids = target_data['frame_id'].values
                    
                    targets_dx.append(dx_vals)
                    targets_dy.append(dy_vals)
                    targets_frame_ids.append(frame_ids)
                else:
                    # Remove sequence if no targets
                    sequences.pop()
                    sequence_ids.pop()
    
    print(f"Created {len(sequences)} sequences")
    
    if is_training:
        return sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids
    else:
        return sequences, sequence_ids

def train_transformer_model_integrated(X_train, y_train, X_val, y_val, input_dim, horizon, config):
    """Integrated training function for Transformer model"""
    device = config.DEVICE
    model = SpatioTemporalTransformer(input_dim, horizon, config).to(device)
    
    criterion = TemporalHuberLoss(delta=0.5, time_decay=0.02, velocity_weight=0.15)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Enhanced scheduler with warm restarts
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS//3, eta_min=1e-6)
    
    # Create datasets and dataloaders
    train_dataset = NFLDataset(X_train, y_train, horizon)
    val_dataset = NFLDataset(X_val, y_val, horizon)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    print(f"Training Transformer with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input dimension: {input_dim}, Horizon: {horizon}")
    
    for epoch in range(1, config.EPOCHS + 1):
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
            
            # Gradient clipping for stability
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
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
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

def main():
    from datetime import datetime
    
    config = TransformerConfig()
    set_seed(config.SEED)
    
    # Load data
    print("\n[1/4] Loading data...")
    train_input_files = [config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
    train_output_files = [config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)]
    train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
    train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
    test_input = pd.read_csv(config.DATA_DIR / "test_input.csv")
    test_template = pd.read_csv(config.DATA_DIR / "test.csv")
    
    # Prepare sequences with Transformer-optimized features
    print("\n[2/4] Preparing sequences with Transformer features...")
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = prepare_sequences_with_transformer_features(
        train_input, train_output, is_training=True, window_size=config.WINDOW_SIZE
    )
    
    sequences = np.array(sequences, dtype=object)
    targets_dx = np.array(targets_dx, dtype=object)
    targets_dy = np.array(targets_dy, dtype=object)
    
    # Cross-validation training
    print("\n[3/4] Training Transformer models...")
    groups = np.array([d['game_id'] for d in sequence_ids])
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    
    models_x, models_y, scalers = [], [], []
    fold_scores = []
    
    for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{config.N_FOLDS}")
        print(f"{'='*60}")
        
        X_tr, X_va = sequences[tr], sequences[va]
        
        # Standardization
        scaler = StandardScaler()
        scaler.fit(np.vstack([s for s in X_tr]))
        
        X_tr_sc = np.stack([scaler.transform(s) for s in X_tr])
        X_va_sc = np.stack([scaler.transform(s) for s in X_va])
        
        input_dim = X_tr[0].shape[-1]
        
        # Train X-axis model
        print("Training X-axis Transformer...")
        mx, loss_x = train_transformer_model_integrated(
            X_tr_sc, targets_dx[tr], X_va_sc, targets_dx[va],
            input_dim, config.MAX_FUTURE_HORIZON, config
        )
        
        # Train Y-axis model
        print("Training Y-axis Transformer...")
        my, loss_y = train_transformer_model_integrated(
            X_tr_sc, targets_dy[tr], X_va_sc, targets_dy[va],
            input_dim, config.MAX_FUTURE_HORIZON, config
        )
        
        models_x.append(mx)
        models_y.append(my)
        scalers.append(scaler)
        
        avg_loss = (loss_x + loss_y) / 2
        fold_scores.append(avg_loss)
        
        print(f"\nFold {fold} - X loss: {loss_x:.5f}, Y loss: {loss_y:.5f}, Avg: {avg_loss:.5f}")
    
    # Print overall performance
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"TRANSFORMER MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Mean CV Score: {mean_score:.5f} ± {std_score:.5f}")
    
    # Test predictions
    print("\n[4/4] Creating test predictions...")
    test_sequences, test_ids = prepare_sequences_with_transformer_features(
        test_input, test_template=test_template, is_training=False, window_size=config.WINDOW_SIZE
    )
    
    X_test = np.array(test_sequences, dtype=object)
    x_last = np.array([s[-1, 0] for s in X_test])
    y_last = np.array([s[-1, 1] for s in X_test])
    
    # Ensemble predictions across folds
    all_dx, all_dy = [], []
    for mx, my, sc in zip(models_x, models_y, scalers):
        X_sc = np.stack([sc.transform(s) for s in X_test])
        X_t = torch.tensor(X_sc.astype(np.float32)).to(config.DEVICE)
        
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
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"submission_transformer_{timestamp}.csv"
    submission.to_csv(submission_file, index=False)
    
    print(f"\nSubmission saved to: {submission_file}")
    print(f"Expected RMSE improvement: 15-20% over baseline (target: 0.58)")
    
    return submission, mean_score

if __name__ == "__main__":
    print("Spatio-Temporal Transformer for NFL Player Movement Prediction")
    print("Integrated pipeline with advanced feature engineering")
    print("="*60)
    
    submission, score = main()
    print(f"\nTransformer pipeline completed with CV score: {score:.5f}")