import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

TARGET = ["x", "y"]

from utilities import prepare_targets_stt

def compute_val_rmse_stt(model, X_val_sc, ydx_list, ydy_list, horizon, device, pm_val=None):
    # stack list -> np.array
    if isinstance(X_val_sc, list):
        X_val_sc = np.stack(X_val_sc).astype(np.float32)

    X_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    PM_t = None
    if pm_val is not None:
        PM_t = torch.tensor(np.stack(pm_val).astype(np.float32)).to(device)

    with torch.no_grad():
        predict = model(X_t, player_mask=PM_t).cpu().numpy()

    # targets & mask
    by, bm = prepare_targets_stt(ydx_list, ydy_list, horizon)
    if torch.is_tensor(by):
        by = by.numpy()
    if torch.is_tensor(bm):
        bm = bm.numpy()

    pdx, pdy = predict[..., 0], predict[..., 1]
    ydx, ydy = by[..., 0], by[..., 1]
    mask = bm

    se_sum2d = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * mask
    denom = mask.sum() + 1e-8

    return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    submission = submission[["id"] + TARGET]
    merged_df = pd.merge(
        solution, submission, on=row_id_column_name, suffixes=("_true", "_pred")
    )
    rmse = np.sqrt(
        0.5
        * (
            mean_squared_error(merged_df["x_true"], merged_df["x_pred"]) +
            mean_squared_error(merged_df["y_true"], merged_df["y_pred"]) 
        )
    )
    return float(rmse)