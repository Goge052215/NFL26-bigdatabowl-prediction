# %%writefile nfl_gnn2.py
# Model 4: GNN Architecture from /kaggle/input/hsiaosuan-sttn/saved_models
# Self-contained version - reuses preprocessing from nfl_mymodel

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import warnings
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# Model Architecture (GRU with Attention Pooling)
# =============================================================================

class GRUSeqModel(nn.Module):
    """
    GRU-based sequence model with attention pooling
    This matches the architecture of models saved in seed_42
    
    Key dimensions from checkpoint:
    - input_dim: 110 (number of input features)
    - in_proj: 110 -> 256
    - GRU: 110 input, 128 hidden, bidirectional (256 output)
    - pool_query: [1, 2, 256] (2 query vectors)
    - After pooling: 2 * 256 = 512
    - head: 512 -> 128 -> 128 -> 94
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, horizon=94):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Input projection: 110 -> 256
        self.in_proj = nn.Linear(110, 256)
        
        # Bidirectional GRU: input=110, hidden=128
        # Note: Takes raw input (110), not projected input
        self.gru = nn.GRU(
            110, hidden_dim, num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention pooling components
        gru_output_dim = hidden_dim * 2  # 256 for bidirectional
        # 2 query vectors for multi-head attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 2, gru_output_dim))
        self.pool_ln = nn.LayerNorm(gru_output_dim)
        self.pool_attn = nn.MultiheadAttention(
            gru_output_dim, num_heads=4, batch_first=True
        )
        
        # Prediction head: 512 (2*256) -> 128 -> 128 -> 94
        # With residual connection via proj layer
        pooled_dim = gru_output_dim * 2  # 512 (concatenated 2 queries)
        self.head = nn.ModuleDict({
            'fc1': nn.Linear(pooled_dim, 128),
            'fc2': nn.Linear(128, 128),
            'proj': nn.Linear(pooled_dim, 128),  # Skip connection
            'out': nn.Linear(128, horizon)
        })
    
    def forward(self, x):
        # x: (B, seq_len, 110)
        # Note: in_proj might not be used in forward pass based on checkpoint
        # GRU takes raw 110-dim input
        h, _ = self.gru(x)  # (B, seq_len, 256)
        
        # Attention pooling with 2 queries
        h_norm = self.pool_ln(h)
        q = self.pool_query.expand(x.size(0), -1, -1)  # (B, 2, 256)
        ctx, _ = self.pool_attn(q, h_norm, h_norm)  # (B, 2, 256)
        ctx = ctx.reshape(x.size(0), -1)  # (B, 512)
        
        # Prediction head with residual connection
        h1 = torch.relu(self.head['fc1'](ctx))  # 512 -> 128
        h2 = torch.relu(self.head['fc2'](h1))   # 128 -> 128
        h3 = torch.relu(self.head['proj'](ctx)) # 512 -> 128 (skip)
        h_combined = h2 + h3                      # Residual connection
        out = self.head['out'](h_combined)       # 128 -> 94
        
        return out

# Import preprocessing from nfl_mymodel (they're in the same directory)
try:
    import importlib
    nfl_mymodel = importlib.import_module('nfl_mymodel')
    
    # Reuse these functions from nfl_mymodel
    prepare_sequences_fixed = nfl_mymodel.prepare_sequences_fixed
    invert_to_original_direction = nfl_mymodel.invert_to_original_direction
    Config = nfl_mymodel.Config
    
    print("[MODEL 4] Reusing preprocessing from nfl_mymodel")
    PREPROCESSING_AVAILABLE = True
except Exception as e:
    print(f"[MODEL 4] Warning: Could not import nfl_mymodel: {e}")
    PREPROCESSING_AVAILABLE = False
    
    # Fallback Config if nfl_mymodel is not available
    class Config:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        WINDOW_SIZE = 12
        MAX_FUTURE_HORIZON = 94
        N_FOLDS = 5
        FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
        FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    def invert_to_original_direction(x_u, y_u, play_dir_right: bool):
        if not play_dir_right:
            return float(x_u), float(y_u)
        return float(120.0 - x_u), float(53.3 - y_u)

# =============================================================================
# Global variables to store models
# =============================================================================
_models_loaded = False
_models_x = None
_models_y = None
_scalers = None
_route_kmeans = None
_route_scaler = None

def load_models_once():
    """Load models on first predict call (no 5-minute time limit)"""
    global _models_loaded, _models_x, _models_y, _scalers, _route_kmeans, _route_scaler
    
    if _models_loaded:
        return
    
    print("[MODEL 4] Loading GNN models for first time...")
    
    # Models are in seed_42 subdirectory
    BASE_DIR = Path("/kaggle/input/hsiaosuan-sttn/saved_models")
    MODEL_DIR = BASE_DIR / "seed_42"
    
    import sys
    sys.stdout.flush()  # Force flush output
    
    try:
        print(f"[MODEL 4 DEBUG] BASE_DIR: {BASE_DIR}", flush=True)
        print(f"[MODEL 4 DEBUG] MODEL_DIR: {MODEL_DIR}", flush=True)
        print(f"[MODEL 4 DEBUG] BASE_DIR exists: {BASE_DIR.exists()}", flush=True)
        print(f"[MODEL 4 DEBUG] MODEL_DIR exists: {MODEL_DIR.exists()}", flush=True)
        
        # List contents to debug
        if BASE_DIR.exists():
            contents = list(BASE_DIR.iterdir())[:10]
            print(f"[MODEL 4 DEBUG] Contents of BASE_DIR: {contents}", flush=True)
        else:
            print(f"[MODEL 4 DEBUG] BASE_DIR does not exist!", flush=True)
        # Load route clustering objects from BASE_DIR (if available)
        try:
            _route_kmeans = joblib.load(BASE_DIR / "route_kmeans.pkl")
            _route_scaler = joblib.load(BASE_DIR / "route_scaler.pkl")
            print("[MODEL 4] Loaded route clustering objects")
        except Exception as e:
            _route_kmeans = None
            _route_scaler = None
            print(f"[MODEL 4] No route clustering objects found (optional): {e}")
        
        # Detect number of folds in seed_42 directory
        print(f"[MODEL 4 DEBUG] Looking for models in: {MODEL_DIR}", flush=True)
        
        if MODEL_DIR.exists():
            # Use iterdir() instead of glob() for better Kaggle compatibility
            all_files = list(MODEL_DIR.iterdir())
            print(f"[MODEL 4 DEBUG] ALL files in MODEL_DIR: {[p.name for p in all_files]}", flush=True)
            
            # Manually filter for model_dx files (glob doesn't work reliably on Kaggle input datasets)
            model_x_paths = sorted([f for f in all_files if f.name.startswith("model_dx_fold") and f.name.endswith(".pt")])
            print(f"[MODEL 4 DEBUG] Found model_dx files: {[p.name for p in model_x_paths]}", flush=True)
        else:
            model_x_paths = []
            print(f"[MODEL 4 DEBUG] MODEL_DIR does not exist, skipping...", flush=True)
        
        n_folds = len(model_x_paths)
        
        if n_folds == 0:
            # Try alternative paths using iterdir() instead of glob()
            print(f"[MODEL 4 DEBUG] No models in seed_42, trying BASE_DIR directly...", flush=True)
            
            if BASE_DIR.exists():
                base_files = list(BASE_DIR.iterdir())
                model_x_paths = sorted([f for f in base_files if f.name.startswith("model_dx_fold") and f.name.endswith(".pt")])
                print(f"[MODEL 4 DEBUG] Found in BASE_DIR: {[p.name for p in model_x_paths]}", flush=True)
            else:
                model_x_paths = []
            
            # Also try searching in subdirectories
            if len(model_x_paths) == 0:
                print(f"[MODEL 4 DEBUG] Trying subdirectories search...", flush=True)
                for seed_dir in BASE_DIR.iterdir():
                    if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                        seed_files = list(seed_dir.iterdir())
                        found = [f for f in seed_files if f.name.startswith("model_dx_fold") and f.name.endswith(".pt")]
                        if found:
                            model_x_paths = sorted(found)
                            MODEL_DIR = seed_dir
                            print(f"[MODEL 4 DEBUG] Found models in {seed_dir.name}: {[p.name for p in model_x_paths]}", flush=True)
                            break
            
            if len(model_x_paths) > 0:
                # If we found files, update MODEL_DIR to the parent directory of the first model
                MODEL_DIR = model_x_paths[0].parent
                n_folds = len(model_x_paths)
                print(f"[MODEL 4] Found {n_folds} models in {MODEL_DIR}", flush=True)
            else:
                error_msg = f"No model files found. Checked:\n"
                error_msg += f"  - {MODEL_DIR} (exists: {MODEL_DIR.exists()})\n"
                error_msg += f"  - {BASE_DIR} (exists: {BASE_DIR.exists()})\n"
                if BASE_DIR.exists():
                    error_msg += f"  - BASE_DIR contents: {[f.name for f in BASE_DIR.iterdir()][:20]}"
                raise FileNotFoundError(error_msg)
        
        print(f"[MODEL 4] Found {n_folds} fold models")
        
        # Load models and scalers
        _models_x = []
        _models_y = []
        _scalers = []
        
        for i in range(1, n_folds + 1):
            model_x_path = MODEL_DIR / f"model_dx_fold{i}.pt"
            model_y_path = MODEL_DIR / f"model_dy_fold{i}.pt"
            scaler_path = MODEL_DIR / f"scaler_fold{i}.pkl"
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            _scalers.append(scaler)
            
            # Store model paths (will load lazily during prediction)
            _models_x.append(model_x_path)
            _models_y.append(model_y_path)
            
            print(f"[MODEL 4] Loaded scaler for fold {i}")
        
        _models_loaded = True
        print(f"[MODEL 4] Successfully loaded {len(_models_x)} models")
        
    except Exception as e:
        print(f"[MODEL 4] Error loading models: {e}")
        raise


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Inference function for Model 4
    
    Args:
        test: Frames to predict
        test_input: Available input data
    
    Returns:
        DataFrame with x, y coordinates
    """
    global _models_x, _models_y, _scalers, _route_kmeans, _route_scaler
    
    # First call: load models (no time limit)
    if not _models_loaded:
        load_models_once()
    
    # Check if preprocessing is available
    if not PREPROCESSING_AVAILABLE:
        print("[MODEL 4] Preprocessing not available, returning default predictions")
        return pl.DataFrame({'x': [60.0] * len(test), 'y': [26.65] * len(test)})
    
    try:
        # Convert to pandas
        test_pd = test.to_pandas()
        test_input_pd = test_input.to_pandas()
        
        print(f"[MODEL 4] Processing {len(test_pd)} predictions...")
        
        # Prepare sequences using nfl_mymodel's preprocessing
        sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y, feature_cols = prepare_sequences_fixed(
            test_input_pd, 
            test_template=test_pd, 
            is_training=False, 
            window_size=Config.WINDOW_SIZE,
            route_kmeans=_route_kmeans,
            route_scaler=_route_scaler
        )
        
        if not sequences:
            print("[MODEL 4] No sequences generated, returning default predictions")
            return pl.DataFrame({'x': [60.0] * len(test), 'y': [26.65] * len(test)})
        
        X_test = np.array(sequences, dtype=object)
        
        # Model 4 was trained on 110 features, but current preprocessing generates more
        # We need to use only the first 110 features to match the trained model
        if X_test[0].shape[1] != 110:
            print(f"[MODEL 4] Feature mismatch: current preprocessing has {X_test[0].shape[1]} features, "
                  f"but model expects 110. Using first 110 features.")
            X_test = np.array([s[:, :110] for s in X_test], dtype=object)
        
        # Extract last positions (unified coordinates)
        x_last_u = np.array([s[-1, 0] for s in X_test])
        y_last_u = np.array([s[-1, 1] for s in X_test])
        
        # Load models and make predictions
        all_dx, all_dy = [], []
        
        for i, (mx_path, my_path, scaler) in enumerate(zip(_models_x, _models_y, _scalers)):
            print(f"[MODEL 4] Processing with fold {i+1}/{len(_models_x)}...")
            
            # Load models for this fold using GRUSeqModel
            # Note: Checkpoint expects input_dim=110, regardless of current preprocessing
            model_x = GRUSeqModel(
                input_dim=110,  # Fixed based on checkpoint
                hidden_dim=128,
                num_layers=2,
                horizon=Config.MAX_FUTURE_HORIZON
            ).to(Config.DEVICE)
            model_x.load_state_dict(torch.load(mx_path, map_location=Config.DEVICE))
            model_x.eval()
            
            model_y = GRUSeqModel(
                input_dim=110,  # Fixed based on checkpoint
                hidden_dim=128,
                num_layers=2,
                horizon=Config.MAX_FUTURE_HORIZON
            ).to(Config.DEVICE)
            model_y.load_state_dict(torch.load(my_path, map_location=Config.DEVICE))
            model_y.eval()
            
            # Scale and predict
            X_scaled = np.stack([scaler.transform(s) for s in X_test])
            X_tensor = torch.tensor(X_scaled.astype(np.float32)).to(Config.DEVICE)
            
            with torch.no_grad():
                dx = model_x(X_tensor).cpu().numpy()
                dy = model_y(X_tensor).cpu().numpy()
            
            all_dx.append(dx)
            all_dy.append(dy)
        
        # Ensemble predictions from all folds
        ens_dx = np.mean(all_dx, axis=0)
        ens_dy = np.mean(all_dy, axis=0)
        
        H = ens_dx.shape[1]
        
        # Build final predictions
        rows = []
        
        for i, sid in enumerate(sequence_ids):
            fids = test_pd[
                (test_pd['game_id'] == sid['game_id']) &
                (test_pd['play_id'] == sid['play_id']) &
                (test_pd['nfl_id'] == sid['nfl_id'])
            ]['frame_id'].sort_values().tolist()
            
            play_dir_right = (sid['play_direction'] == 'right')
            
            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                
                # Calculate unified coordinates
                x_u = np.clip(x_last_u[i] + ens_dx[i, tt], 0, Config.FIELD_X_MAX)
                y_u = np.clip(y_last_u[i] + ens_dy[i, tt], 0, Config.FIELD_Y_MAX)
                
                # Convert back to original direction
                x_orig, y_orig = invert_to_original_direction(x_u, y_u, play_dir_right)
                
                rows.append({
                    'x': float(x_orig),
                    'y': float(y_orig)
                })
        
        predictions = pl.DataFrame(rows)
        
        if len(predictions) != len(test):
            print(f"[MODEL 4 WARNING] Prediction count mismatch: {len(predictions)} vs {len(test)}")
            # Pad or truncate to match expected length
            if len(predictions) < len(test):
                missing = len(test) - len(predictions)
                padding = pl.DataFrame({'x': [60.0] * missing, 'y': [26.65] * missing})
                predictions = pl.concat([predictions, padding])
            else:
                predictions = predictions.head(len(test))
        
        print(f"[MODEL 4] Completed {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        print(f"[MODEL 4 ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return default predictions as fallback
        print("[MODEL 4] Returning default predictions as fallback")
        return pl.DataFrame({'x': [60.0] * len(test), 'y': [26.65] * len(test)})
