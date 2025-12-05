import time
import numpy as np
import torch
import torch.nn as nn
from config import Config
from model import STTransformer, TemporalHuber
from utilities import prepare_targets_stt

def train_model_stt(
    X_train,
    y_train_dx,
    y_train_dy,
    X_val,
    y_val_dx,
    y_val_dy,
    input_dim,
):
    device = Config.DEVICE
    print(f"[Device] Training on {device}")

    # Construct train/val dataset
    train_batches = []
    for i in range(0, len(X_train), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets_stt(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            Config.MAX_FUTURE_HORIZON,
        )
        train_batches.append((bx, by, bm))

    val_batches = []
    for i in range(0, len(X_val), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets_stt(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            Config.MAX_FUTURE_HORIZON,
        )
        val_batches.append((bx, by, bm))

    # Define model, criterion, optimizer, scheduler
    model = STTransformer(
        input_dim=input_dim,
    ).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5
    )
    # total_steps = Config.EPOCHS * len(train_batches)
    # warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_loss, best_state, bad = float("inf"), None, 0
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
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
            # scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        # Accumulate squared error for RMSE calculation across all validation batches
        se_sum = 0.0
        denom_sum = 0.0
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())
                # Compute RMSE components on this batch
                pdx, pdy = pred[..., 0], pred[..., 1]
                ydx, ydy = by[..., 0], by[..., 1]
                mask = bm
                se_batch = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * mask
                se_sum += float(se_batch.sum().item())
                denom_sum += float(mask.sum().item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)

        rmse_val = float(np.sqrt(se_sum / (2.0 * (denom_sum + 1e-8)))) if denom_sum > 0 else float('nan')

        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(
            f"  Epoch {epoch:>3}: train={train_loss:.4f}, val={val_loss:.4f}, rmse={rmse_val:.4f}, "
            f"Time_elapsed={minutes:>2}min {seconds:>2}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= Config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        # best epoch, 根据这个epoch再跑全数据 -> pth, pkl save
        #

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss