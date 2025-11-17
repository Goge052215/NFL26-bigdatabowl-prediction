import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue

from config import Config
from utilities import (
    build_play_direction_map,
    unify_left_direction_ipt,
    unify_left_direction_opt,
)
from feature import FeatureEngineer

def _canonicalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("game_id", "play_id", "nfl_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Handle missing keys
    df = df.dropna(subset=["game_id", "play_id", "nfl_id"])
    # Convert to int64?
    df["game_id"] = df["game_id"].astype("int64")
    df["play_id"] = df["play_id"].astype("int64")
    df["nfl_id"] = df["nfl_id"].astype("int64")
    return df

def _process_group_batch(
    batch_keys: list,
    grouped_dict: dict,
    feature_cols: list,
    target_rows: pd.DataFrame,
    idx_x: int,
    idx_y: int,
    dir_map: pd.DataFrame,
    queue: Queue,
):
    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []
    for key in batch_keys:
        gid, pid, nid = key
        group_df = grouped_dict.get(key)
        if group_df is None:
            continue

        # Build input window
        input_window = group_df.tail(Config.WINDOW_SIZE)
        if len(input_window) < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - len(input_window)
            pad_df = pd.DataFrame(
                np.nan, index=range(pad_len), columns=input_window.columns
            )
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        input_window = input_window.fillna(input_window.mean(numeric_only=True))
        seq = input_window[feature_cols].to_numpy(dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        sequences.append(seq)

        # Training targets
        if Config.TRAIN:
            out_grp: pd.DataFrame = target_rows[
                (target_rows["game_id"] == gid)
                & (target_rows["play_id"] == pid)
                & (target_rows["nfl_id"] == nid)
            ].sort_values("frame_id")
            if len(out_grp) == 0:
                sequences.pop()
                continue
            dx = out_grp["x"].to_numpy(np.float32) - seq[-1, idx_x]
            dy = out_grp["y"].to_numpy(np.float32) - seq[-1, idx_y]
            fids = out_grp["frame_id"].to_numpy(np.int32)
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_fids.append(fids)

        play_dir_val = dir_map.loc[(gid, pid)]
        seq_meta.append(
            {
                "game_id": gid,
                "play_id": pid,
                "nfl_id": nid,
                "frame_id": int(input_window.iloc[-1]["frame_id"]),
                "play_direction": play_dir_val,
            }
        )

        if queue is not None:
            queue.put(1)

    return sequences, targets_dx, targets_dy, targets_fids, seq_meta

def prepare_sequences_with_advanced_features(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_groups: list = None,
    required_feature_cols: list | None = None,
):

    print(f"\n{'='*80}")
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES (UNIFIED FRAME)")
    print(f"{'='*80}")
    print(f"Window size: {Config.WINDOW_SIZE}")

    input_df = _canonicalize_key_dtypes(input_df)
    output_df = _canonicalize_key_dtypes(output_df)

    dir_map = build_play_direction_map(input_df)
    input_df = unify_left_direction_ipt(input_df)
    output_df = unify_left_direction_opt(output_df, dir_map)

    target_rows = output_df
    target_groups = output_df[["game_id", "play_id", "nfl_id"]].drop_duplicates()

    # Feature Engineering
    fe = FeatureEngineer(feature_groups)
    processed_df, feature_cols = fe.transform(input_df)

    if required_feature_cols is not None:
        req = list(required_feature_cols)
        missing = [c for c in req if c not in processed_df.columns]
        for c in missing:
            processed_df[c] = 0.0
        feature_cols = req

    # Build sequences
    start_time = time.time()
    grouped_dict = {
        (gid, pid, nid): g
        for (gid, pid, nid), g in processed_df.groupby(
            ["game_id", "play_id", "nfl_id"], sort=False
        )
    }

    # Check if "x" and "y" are in feature_cols but not available in the original form
    if "x" in feature_cols and "x" not in input_df.columns:
        # "x" is required but not available - this should not happen
        raise ValueError("Required feature 'x' is not available in input data")
    if "y" in feature_cols and "y" not in input_df.columns:
        # "y" is required but not available - this should not happen
        raise ValueError("Required feature 'y' is not available in input data")

    # helpful indices
    idx_x = feature_cols.index("x") if "x" in feature_cols else None
    idx_y = feature_cols.index("y") if "y" in feature_cols else None

    # Spread group across cpus
    all_keys = [tuple(x) for x in target_groups.to_numpy()]
    batch_size = (len(all_keys) + Config.MAX_WORKER - 1) // Config.MAX_WORKER
    batches = [
        all_keys[i : i + batch_size] for i in range(0, len(all_keys), batch_size)
    ]

    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []

    if Config.TRAIN:
        manager = Manager()
        queue = manager.Queue()
        pbar = tqdm(total=len(all_keys), desc="Creating sequences (groups)")

        # Build sequences in parallel
        with ProcessPoolExecutor(max_workers=Config.MAX_WORKER) as ex:
            futures = [
                ex.submit(
                    _process_group_batch,
                    b,
                    grouped_dict,
                    feature_cols,
                    target_rows,
                    idx_x,
                    idx_y,
                    dir_map,
                    queue,
                )
                for b in batches
            ]
            finished = 0
            while finished < len(all_keys):
                queue.get()
                finished += 1
                pbar.update(1)

            # Wait for all task to complete
            for fut in as_completed(futures):
                seqs, dxs, dys, fids_list, metas = fut.result()
                sequences.extend(seqs)
                targets_dx.extend(dxs)
                targets_dy.extend(dys)
                targets_fids.extend(fids_list)
                seq_meta.extend(metas)

        pbar.close()

    else:
        # No multiprocessing when not training
        print("[INFO] Running in single-process mode")
        pbar = tqdm(total=len(all_keys), desc="Creating sequences (groups)")
        for key in all_keys:
            seqs, dxs, dys, fids_list, metas = _process_group_batch(
                [key],
                grouped_dict,
                feature_cols,
                target_rows,
                idx_x,
                idx_y,
                dir_map,
                None,
            )
            sequences.extend(seqs)
            seq_meta.extend(metas)
            pbar.update(1)
        pbar.close()
    end_time = time.time()
    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")
    print(f"Time to build sequences: {end_time - start_time:.2f} seconds")

    if Config.TRAIN:
        return (
            sequences,
            targets_dx,
            targets_dy,
            targets_fids,
            seq_meta,
            feature_cols,
        )
    return sequences, seq_meta, feature_cols

def _order_players_for_play(play_df: pd.DataFrame, target_nfl_id: int, end_frame: int) -> list:
    last = (
        play_df[play_df["frame_id"] == end_frame]
        .groupby("nfl_id", as_index=False)[["x", "y", "player_side"]]
        .first()
    )
    if target_nfl_id not in set(last["nfl_id"].tolist()):
        target_nfl_id = int(play_df["nfl_id"].iloc[0])
    tx = float(last.loc[last["nfl_id"] == target_nfl_id, "x"].values[0])
    ty = float(last.loc[last["nfl_id"] == target_nfl_id, "y"].values[0])
    last["dist2target"] = np.hypot(last["x"] - tx, last["y"] - ty)
    other_ids = last[last["nfl_id"] != target_nfl_id].sort_values("dist2target")[
        "nfl_id"
    ].tolist()
    ordered = [target_nfl_id] + other_ids
    return ordered

def prepare_sequences_play_level(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_groups: list = None,
    required_feature_cols: list | None = None,
):
    input_df = _canonicalize_key_dtypes(input_df)
    output_df = _canonicalize_key_dtypes(output_df)

    dir_map = build_play_direction_map(input_df)
    input_df = unify_left_direction_ipt(input_df)
    output_df = unify_left_direction_opt(output_df, dir_map)

    fe = FeatureEngineer(feature_groups)
    processed_df, base_feature_cols = fe.transform(input_df)

    if required_feature_cols is not None:
        req = list(required_feature_cols)
        missing = [c for c in req if c not in processed_df.columns]
        for c in missing:
            processed_df[c] = 0.0
        base_feature_cols = req

    # Check if "x" and "y" are in base_feature_cols but not available in the original form
    # If they're missing from the processed data but required, we need to handle this
    if "x" in base_feature_cols and "x" not in input_df.columns:
        # "x" is required but not available - this should not happen
        raise ValueError("Required feature 'x' is not available in input data")
    if "y" in base_feature_cols and "y" not in input_df.columns:
        # "y" is required but not available - this should not happen
        raise ValueError("Required feature 'y' is not available in input data")

    # Use the original input data for position access since feature engineering may exclude raw positions
    idx_x = base_feature_cols.index("x") if "x" in base_feature_cols else None
    idx_y = base_feature_cols.index("y") if "y" in base_feature_cols else None

    sequences, player_masks, targets_dx, targets_dy, targets_fids, seq_meta = (
        [], [], [], [], [], []
    )

    for (gid, pid), play_df in processed_df.groupby(["game_id", "play_id"], sort=False):
        play_df = play_df.sort_values(["nfl_id", "frame_id"]).reset_index(drop=True)
        end_frame = int(play_df["frame_id"].max())
        start_frame = max(end_frame - Config.WINDOW_SIZE + 1, int(play_df["frame_id"].min()))

        tr_ids = (
            play_df.loc[play_df["player_role"] == "Targeted Receiver", "nfl_id"].unique()
        )
        if len(tr_ids) == 0:
            continue
        target_nfl_id = int(tr_ids[0])

        ordered_ids = _order_players_for_play(play_df, target_nfl_id, end_frame)
        selected_ids = ordered_ids[: Config.MAX_PLAYER]

        P = Config.MAX_PLAYER
        T = Config.WINDOW_SIZE
        D_base = len(base_feature_cols)
        play_tensor = np.zeros((P, T, D_base), dtype=np.float32)
        player_mask = np.zeros((P,), dtype=np.float32)

        ref_x, ref_y = None, None

        for p_idx in range(P):
            if p_idx >= len(selected_ids):
                continue
            nid = selected_ids[p_idx]
            sub = play_df[(play_df["nfl_id"] == nid) & (play_df["frame_id"].between(start_frame, end_frame))]
            if len(sub) < T:
                pad_len = T - len(sub)
                pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=sub.columns)
                sub = pd.concat([pad_df, sub], ignore_index=True)

            sub = sub.fillna(sub.mean(numeric_only=True))
            seq = sub[base_feature_cols].to_numpy(dtype=np.float32)
            seq = np.nan_to_num(seq, nan=0.0)

            play_tensor[p_idx] = seq[-T:]
            player_mask[p_idx] = 1.0

            if nid == target_nfl_id:
                ref_x = float(seq[-1, idx_x])
                ref_y = float(seq[-1, idx_y])

        if ref_x is None or ref_y is None:
            continue

        out_grp = (
            output_df[(output_df["game_id"] == gid) & (output_df["play_id"] == pid) & (output_df["nfl_id"] == target_nfl_id)]
            .sort_values("frame_id")
        )
        if len(out_grp) == 0:
            continue
        dx = out_grp["x"].to_numpy(np.float32) - np.float32(ref_x)
        dy = out_grp["y"].to_numpy(np.float32) - np.float32(ref_y)
        fids = out_grp["frame_id"].to_numpy(np.int32)

        try:
            sides_last = (
                play_df[play_df["frame_id"] == end_frame]
                .groupby("nfl_id", as_index=False)["player_side"]
                .first()
            )
            side_map = {int(r["nfl_id"]): str(r["player_side"]) for _, r in sides_last.iterrows()}
            sides = [side_map.get(int(nid), "Offense") for nid in selected_ids]

            X = play_tensor[:, :, idx_x]
            Y = play_tensor[:, :, idx_y]
            tv_pressure = np.zeros((P, T), dtype=np.float32)
            tv_ally_density = np.zeros((P, T), dtype=np.float32)
            tv_oppn_density = np.zeros((P, T), dtype=np.float32)
            tv_density_ratio = np.zeros((P, T), dtype=np.float32)
            tv_dist_min = np.zeros((P, T), dtype=np.float32)

            R = float(getattr(Config, "RADIUS", 30.0))
            for t in range(T):
                valid_idx = [i for i in range(P) if player_mask[i] > 0.5]
                if len(valid_idx) <= 1:
                    continue
                coords = np.stack([X[valid_idx, t], Y[valid_idx, t]], axis=-1)
                for loc_pos, i in enumerate(valid_idx):
                    xi, yi = coords[loc_pos]
                    diffs = coords - np.array([xi, yi], dtype=np.float32)
                    dists = np.sqrt((diffs ** 2).sum(axis=-1) + 1e-8)
                    dists[loc_pos] = np.inf
                    tv_dist_min[i, t] = np.min(dists)

                    si = sides[loc_pos]
                    ally_mask = np.array([1.0 if sides[j] == si else 0.0 for j in range(len(valid_idx))], dtype=np.float32)
                    ally_mask[loc_pos] = 0.0
                    opp_mask = 1.0 - ally_mask

                    ally_count = float(np.sum((dists <= R) * ally_mask))
                    opp_count = float(np.sum((dists <= R) * opp_mask))
                    area = np.pi * (R ** 2)
                    tv_ally_density[i, t] = ally_count / (area + 1e-6)
                    tv_oppn_density[i, t] = opp_count / (area + 1e-6)
                    tv_density_ratio[i, t] = tv_ally_density[i, t] / (tv_oppn_density[i, t] + 1e-6)

                    opp_dists = dists[opp_mask > 0.5]
                    opp_near = float(np.min(opp_dists)) if opp_dists.size > 0 else np.inf
                    tv_pressure[i, t] = 1.0 / max(opp_near, 0.5)

            new_feats = np.stack(
                [tv_pressure, tv_ally_density, tv_oppn_density, tv_density_ratio, tv_dist_min],
                axis=-1,
            )
            play_tensor = np.concatenate([play_tensor, new_feats], axis=-1)
        except Exception:
            zeros_feats = np.zeros((P, T, 5), dtype=np.float32)
            play_tensor = np.concatenate([play_tensor, zeros_feats], axis=-1)

        if getattr(Config, "ADD_BALL_TOKEN", False):
            try:
                bx_val = float(play_df["ball_land_x"].dropna().iloc[-1])
                by_val = float(play_df["ball_land_y"].dropna().iloc[-1])
            except Exception:
                bx_val = float(play_df["x"].iloc[-1])
                by_val = float(play_df["y"].iloc[-1])

            D_total = int(play_tensor.shape[-1])
            ball_seq = np.zeros((T, D_total), dtype=np.float32)
            bx_i = base_feature_cols.index("ball_land_x") if "ball_land_x" in base_feature_cols else None
            by_i = base_feature_cols.index("ball_land_y") if "ball_land_y" in base_feature_cols else None
            x_i = idx_x
            y_i = idx_y
            ball_seq[:, x_i] = bx_val
            ball_seq[:, y_i] = by_val
            if bx_i is not None:
                ball_seq[:, bx_i] = bx_val
            if by_i is not None:
                ball_seq[:, by_i] = by_val

            play_tensor = np.concatenate([play_tensor, ball_seq[np.newaxis, :, :]], axis=0)
            player_mask = np.concatenate([player_mask, np.array([1.0], dtype=np.float32)], axis=0)

        sequences.append(play_tensor)
        player_masks.append(player_mask)
        targets_dx.append(dx)
        targets_dy.append(dy)
        targets_fids.append(fids)
        seq_meta.append({"game_id": gid, "play_id": pid, "nfl_id": target_nfl_id})

    extra_cols = [
        "tv_pressure",
        "tv_ally_density",
        "tv_oppn_density",
        "tv_density_ratio",
        "tv_dist_min",
    ]
    feature_cols_total = base_feature_cols + extra_cols

    if Config.TRAIN:
        return (
            sequences,
            targets_dx,
            targets_dy,
            targets_fids,
            seq_meta,
            feature_cols_total,
            player_masks,
        )
    return sequences, seq_meta, feature_cols_total, player_masks