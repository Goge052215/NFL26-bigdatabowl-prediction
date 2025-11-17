## Training Pipeline

### 1. Config

#### 1.1 Feature Groups

Feature groups in the training process: 
```python
    FEATURE_GROUPS = [
        "target_alignment", "lag", "motion_change",
        "field_position"," distance_rate", "geometric", 
        "neighbor_gnn", "time", "passer", "curvature",
        "route", "receiver",
    ]
```

The feature groups above have the similar portrait of the game play. In NFL, we are often measuring the prediction of the player's trajectory based on a whole series of the ball's movement. Hence, straightforward features such as `time`, `passer`, `receiver`, etc. give us the first-hand information of a specific player in a game.

Furthermore, applying:
- GNN (Graph Neural Network) to the feature group `neighbor_gnn` can help us capture the spatial relationship between players.
- Physical calculation to the feature group `motion_change` can help us measure the player's acceleration and deceleration.
- Geometric calculation to the feature group `geometric` can help us measure the player's distance to the ball and the ball's distance to the receiver.

#### 1.2 Parameters Selection

**GNN Parameters**
- `K_NEIGH` : $3$
- `RADIUS` : $r = 28.0$
- `TAU` : $\tau = 8$

