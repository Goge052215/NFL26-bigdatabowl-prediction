import torch
import numpy as np

# TODO: TTA?
def predict_sst(model, scaler, X_test_raw, device, pm_test=None):
    model.eval()
    outs_dx, outs_dy = [], []

    if X_test_raw and np.array(X_test_raw[0]).ndim == 3:
        base = np.stack([
            scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape)
            for s in X_test_raw
        ]).astype(np.float32)
    else:
        base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)
    pm_t = None
    if pm_test is not None:
        pm_t = torch.tensor(np.stack(pm_test).astype(np.float32), device=device)

    with torch.no_grad():
        output = model(xt, player_mask=pm_t)

        dx = output[:, :, 0]  # 第一维为 dx
        dy = output[:, :, 1]  # 第二维为 dy

    outs_dx.append(dx.detach().cpu().numpy())
    outs_dy.append(dy.detach().cpu().numpy())

    return np.mean(outs_dx, axis=0), np.mean(outs_dy, axis=0)