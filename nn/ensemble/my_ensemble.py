# --- ENSEMBLE: Two GNN models + One GRU model ---
# Model 1 (GNN): /kaggle/input/1103new-all-all-all/pytorch/default/1/1103new_all_all_all
# Model 2 (GRU): /kaggle/input/1114kaiyuan-0562/output/20251108_071543
# Model 4 (GNN): /kaggle/input/hsiaosuan-sttn/saved_models/seed_42

import importlib
import polars as pl
import pandas as pd
import kaggle_evaluation.nfl_inference_server
import os

# Import the three model modules
# Model 1: nfl_mymodel (GNN-based architecture)
# Model 2: nfl_gru (GRU/Transformer-based architecture)  
# Model 4: nfl_gnn2 (GNN-based architecture)
nfl_mymodel = importlib.import_module('nfl_mymodel')  # Model 1
nfl_gru = importlib.import_module('nfl_gru')          # Model 2
nfl_gnn2 = importlib.import_module('nfl_gnn2')        # Model 4

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
    """
    Ensemble prediction using three models:
    - Model 1 (nfl_mymodel): GNN architecture
    - Model 2 (nfl_gru): GRU/Transformer architecture
    - Model 4 (nfl_gnn2): GNN architecture
    """
    
    # Get predictions from Model 1 (GNN)
    print("[ENSEMBLE] Getting predictions from Model 1 (nfl_mymodel)...")
    pred_mymodel = nfl_mymodel.predict(test, test_input)
    
    # Get predictions from Model 2 (GRU)
    print("[ENSEMBLE] Getting predictions from Model 2 (nfl_gru)...")
    pred_gru = nfl_gru.predict(test, test_input)
    
    # Get predictions from Model 4 (GNN)
    print("[ENSEMBLE] Getting predictions from Model 4 (nfl_gnn2)...")
    pred_gnn2 = nfl_gnn2.predict(test, test_input)
    
    # Convert all predictions to Pandas DataFrames for ensembling
    if isinstance(pred_mymodel, pl.DataFrame):
        pred_mymodel = pred_mymodel.to_pandas()
    if isinstance(pred_gru, pl.DataFrame):
        pred_gru = pred_gru.to_pandas()
    if isinstance(pred_gnn2, pl.DataFrame):
        pred_gnn2 = pred_gnn2.to_pandas()
    
    # Ensemble predictions with weighted average
    # Using equal weights (0.33, 0.34, 0.33) for the three models
    pred_ensemble = (
        pred_mymodel[['x', 'y']].values * 0.33 + 
        pred_gru[['x', 'y']].values * 0.34 + 
        pred_gnn2[['x', 'y']].values * 0.33
    )
    
    print("[ENSEMBLE] Ensemble complete.")
    
    # Return final ensemble predictions (must contain only 'x' and 'y' columns)
    return pd.DataFrame(pred_ensemble, columns=['x', 'y'])

# Setup inference server with the new ensemble predict function
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))