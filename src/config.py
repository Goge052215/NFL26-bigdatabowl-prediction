import os
import torch
from pathlib import Path

def is_kaggle():
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

class Config:
    TIME_TAG = "default"

    # Status flag
    TRAIN = not is_kaggle()
    LOCAL = True
    SUBMIT = not TRAIN

    # Debug mode: check pipeline integrity
    DEBUG = True
    DEBUG_SIZE = 1

    PREFIX = "/kaggle/input/"
    DATA_DIR = Path("data") if LOCAL else Path(f"{PREFIX}nfl-big-data-bowl-2026-prediction/")
    OUTPUT_DIR = Path("./output")
    SAVE_DIR = Path(f"./output/{TIME_TAG}")

    # fallback to a single process in submit mode
    MAX_WORKER = min(8, os.cpu_count() or 1) if TRAIN else 1

    # Specify the feature group
    FEATURE_GROUPS = [
        "target_alignment",
        "lag",
        "motion_change",
        "field_position",
        "distance_rate",
        "geometric",
        "neighbor_gnn",
        "time",
        "passer",
        "curvature",
        "route",
        "receiver",
    ]

    # Neighbors feature
    K_NEIGH = 3
    RADIUS = 28.0
    TAU = 8.0

    # Training Setting
    SEEDS = [42]
    # SEEDS = [42, 19, 89, 64]
    N_FOLDS = 5
    BATCH_SIZE = 256
    EPOCHS = 200 if not DEBUG else 20
    PATIENCE = 30
    LEARNING_RATE = 1e-3

    WINDOW_SIZE = 10
    MAX_PLAYER = 9
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 55  # Number of steps to predict (filter out 94)

    N_HEADS = 4
    N_LAYERS = 2
    MLP_HIDDEN_DIM = 256
    N_RES_BLOCKS = 2
    N_QUERYS = 3

    # Input/Model options
    USE_PLAY_PLAYER_INPUT = True
    ADD_BALL_TOKEN = True
    USE_AXIAL_ATTENTION = True

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # Field Setting
    YARDS_TO_METERS = 0.9144
    FPS = 10.0

    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3