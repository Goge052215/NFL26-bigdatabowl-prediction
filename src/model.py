import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import get_cosine_schedule_with_warmup

from config import Config
from utilities import (
    prepare_targets_stt,
    save_fold_artifacts_stt,
)
from validation import compute_val_rmse_stt


class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.02, lam_smooth=0.01):
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
            self.delta * (abs_err - 0.5 * self.delta),
        )

        # time decay
        if self.time_decay and self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device, dtype=pred.dtype)
            w = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * w
            mask = mask.unsqueeze(-1) * w

        main_loss = (huber * mask).sum() / (mask.sum() + 1e-8)

        # # velocity smooth
        # if self.lam_smooth and pred.size(1) > 2:
        #     d1 = pred[:, 1:] - pred[:, :-1]
        #     d2 = d1[:, 1:] - d1[:, :-1]
        #     m2 = mask[:, 2:]
        #     smooth = (d2 * d2) * m2
        #     smooth_loss = smooth.sum() / (m2.sum() + 1e-8)
        # else:
        #     smooth_loss = pred.new_tensor(0.0)

        return main_loss


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_pos = max_position

    def forward(self, x):
        t = torch.arange(x.size(1), device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        sin, cos = freqs.sin(), freqs.cos()
        sin = torch.repeat_interleave(sin, 2, dim=1)
        cos = torch.repeat_interleave(cos, 2, dim=1)
        return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(x) + x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # 隐藏层（带残差连接）
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class STTransformer(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.horizon = Config.MAX_FUTURE_HORIZON
        self.hidden_dim = Config.HIDDEN_DIM
        self.n_heads = Config.N_HEADS
        self.n_layers = Config.N_LAYERS
        self.n_querys = Config.N_QUERYS

        self.input_projection = nn.Linear(input_dim, self.hidden_dim)

        self.pos_time = nn.Parameter(torch.randn(1, Config.WINDOW_SIZE, self.hidden_dim))
        self.pos_player = nn.Parameter(
            torch.randn(
                1,
                Config.MAX_PLAYER + (1 if getattr(Config, "ADD_BALL_TOKEN", False) else 0),
                self.hidden_dim,
            )
        )
        self.embed_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)

        time_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        player_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=self.n_layers)
        self.player_encoder = nn.TransformerEncoder(player_layer, num_layers=self.n_layers)

        self.pool_ln = nn.LayerNorm(self.hidden_dim)

        self.head = ResidualMLP(
            input_dim=self.n_querys * self.hidden_dim,
            hidden_dim=Config.MLP_HIDDEN_DIM,
            output_dim=self.horizon * 2,
            num_layers=Config.N_RES_BLOCKS,
            dropout=0.2,
        )

    def forward(self, x: torch.Tensor, player_mask: torch.Tensor = None):
        if x.dim() == 3:
            B, T, _ = x.shape
            x_embed = self.input_projection(x)
            x_embed = x_embed + self.pos_time[:, :T, :]
            x_embed = self.embed_dropout(x_embed)
            h = self.transformer_encoder(x_embed)
            h = self.pool_ln(h)
            ctx = h[:, -1, :]
            if self.n_querys > 1:
                ctx = torch.cat([ctx] * self.n_querys, dim=-1)
            out = self.head(ctx)
            out = out.view(B, self.horizon, 2)
            out = torch.cumsum(out, dim=1)
            return out
        elif x.dim() == 4:
            B, P, T, D = x.shape
            x_proj = self.input_projection(x)
            pos_t = self.pos_time[:, :T, :].unsqueeze(1)
            if getattr(Config, "USE_AXIAL_ATTENTION", False):
                x_t = self.embed_dropout(x_proj + pos_t)
                h_t = self.time_encoder(x_t.reshape(B * P, T, self.hidden_dim))
                h_t = h_t.reshape(B, P, T, self.hidden_dim)
                h_last = h_t[:, :, T - 1, :]
                pos_p = self.pos_player[:, :P, :]
                h_players = self.embed_dropout(h_last + pos_p)
                pad_players = None
                if player_mask is not None and player_mask.dim() == 2:
                    pad_players = (player_mask == 0)
                h_players = self.player_encoder(h_players, src_key_padding_mask=pad_players)
                h_players = self.pool_ln(h_players)
                q_idxs = [0]
                if getattr(Config, "ADD_BALL_TOKEN", False) and P >= (Config.MAX_PLAYER + 1):
                    q_idxs.append(P - 1)
                if len(q_idxs) < self.n_querys and P > 1:
                    q_idxs.append(1)
                q_idxs = q_idxs[: self.n_querys]
                ctx = torch.cat([h_players[:, qi, :] for qi in q_idxs], dim=-1)
            else:
                pos_p = self.pos_player[:, :P, :].unsqueeze(2)
                x_embed = self.embed_dropout(x_proj + pos_t + pos_p)
                x_tok = x_embed.reshape(B, P * T, self.hidden_dim)
                pad_mask = None
                if player_mask is not None:
                    if player_mask.dim() == 2:
                        pm = player_mask.unsqueeze(-1).expand(B, P, T)
                    else:
                        pm = player_mask
                    pad_mask = (pm.reshape(B, P * T) == 0)
                h = self.transformer_encoder(x_tok, src_key_padding_mask=pad_mask)
                h = self.pool_ln(h)
                ctx = h[:, T - 1, :]
            out = self.head(ctx)
            out = out.view(B, self.horizon, 2)
            out = torch.cumsum(out, dim=1)
            return out
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


def train_model_stt(
    X_train,
    y_train_dx,
    y_train_dy,
    X_val,
    y_val_dx,
    y_val_dy,
    input_dim,
    pm_train=None,
    pm_val=None,
):
    device = Config.DEVICE

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
        pm = None
        if pm_train is not None:
            pm = torch.tensor(np.stack(pm_train[i:end]).astype(np.float32))
        train_batches.append((bx, by, bm, pm))

    val_batches = []
    for i in range(0, len(X_val), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets_stt(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            Config.MAX_FUTURE_HORIZON,
        )
        pm = None
        if pm_val is not None:
            pm = torch.tensor(np.stack(pm_val[i:end]).astype(np.float32))
        val_batches.append((bx, by, bm, pm))

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
        for bx, by, bm, pm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pm = pm.to(device) if pm is not None else None
            pred = model(bx, player_mask=pm)
            loss = criterion(pred, by, bm)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm, pm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pm = pm.to(device) if pm is not None else None
                pred = model(bx, player_mask=pm)
                val_losses.append(criterion(pred, by, bm).item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            total_time = time.time() - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(
                f"  Epoch {epoch:>3}: train={train_loss:.4f}, val={val_loss:.4f}, "
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

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


def train_all_folds_stt(
    gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim, player_masks=None
):
    fold_rmses = []
    all_rmse = []
    cv_log = []

    for fold, (tr, va) in enumerate(gkf.split(sequences, y=None, groups=groups), 1):
        print(f"\n{'-'*60}\nFold {fold}/{Config.N_FOLDS} (seed {seed})\n{'-'*60}")

        X_tr = [sequences[i] for i in tr]
        X_va = [sequences[i] for i in va]
        y_tr_dx = [targets_dx[i] for i in tr]
        y_va_dx = [targets_dx[i] for i in va]
        y_tr_dy = [targets_dy[i] for i in tr]
        y_va_dy = [targets_dy[i] for i in va]

        scaler = StandardScaler()
        if X_tr and np.array(X_tr[0]).ndim == 3:
            scaler.fit(np.vstack([s.reshape(-1, s.shape[-1]) for s in X_tr]))
            X_tr_sc = [scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape) for s in X_tr]
            X_va_sc = [scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape) for s in X_va]
        else:
            scaler.fit(np.vstack([s for s in X_tr]))
            X_tr_sc = [scaler.transform(s) for s in X_tr]
            X_va_sc = [scaler.transform(s) for s in X_va]

        PM_tr = [player_masks[i] for i in tr] if player_masks is not None else None
        PM_va = [player_masks[i] for i in va] if player_masks is not None else None
        model, loss = train_model_stt(
            X_tr_sc,
            y_tr_dx,
            y_tr_dy,
            X_va_sc,
            y_va_dx,
            y_va_dy,
            input_dim,
            pm_train=PM_tr,
            pm_val=PM_va,
        )

        rmse = compute_val_rmse_stt(
            model,
            X_va_sc,
            [targets_dx[i] for i in va],
            [targets_dy[i] for i in va],
            Config.MAX_FUTURE_HORIZON,
            Config.DEVICE,
            pm_val=PM_va,
        )

        print(
            f"[VAL] seed {seed} fold {fold} → "
            f"Huber loss={loss:.5f} | "
            f"RMSE={rmse:.4f}"
        )

        fold_rmses.append(rmse)
        all_rmse.append(rmse)
        cv_log.append(
            {
                "seed": seed,
                "fold": fold,
                "rmse": rmse,
                "loss": float(loss),
            }
        )

        # Save model
        save_fold_artifacts_stt(
            seed=seed,
            fold=fold,
            scaler=scaler,
            model=model,
            base_dir=Config.SAVE_DIR,
        )

    print(
        f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
        f"mean={float(np.mean(fold_rmses)):.4f} yards"
    )

    return all_rmse, cv_log