import gc, time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Manager, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import Config
from utilities import wrap_angle_deg

class FeatureEngineer:
    def __init__(self, feature_groups_to_create: list):
        self.gcols = ["game_id", "play_id", "nfl_id"]
        self.active_groups = feature_groups_to_create
        self.feature_creators = {
            "target_alignment": (self._create_target_alignment_features, False),
            "multi_window": (self._create_multi_window_features, False),
            "lag": (self._create_extended_lag_features, False),
            "motion_change": (self._create_motion_change_features, False),
            "field_position": (self._create_field_position_features, False),
            "distance_rate": (self._create_distance_rate_features, False),
            "geometric": (self._create_geometric_features, False),
            "neighbor_gnn": (self._create_neighbor_features, True),
            "time": (self._create_time_features, False),
            "role": (self._create_role_features, False),
            "passer": (self._create_passer_features, True),
            "curvature": (self._create_curvature_features, False),
            "route": (self._create_route_features, False),
            "receiver": (self._create_receiver_features, True),
        }
        self.created_feature_cols = []

    def _height_to_feet(self, height_str):
        try:
            ft, inches = map(int, str(height_str).split("-"))
            return ft + inches / 12
        except Exception:
            return 6.0

    def _mirror_angle(self, df: pd.DataFrame, cols: list):
        for col in cols:
            df[col] = (450 - df[col]) % 360
        return df

    def _warp_angle(self, df: pd.DataFrame, col: str):
        return np.minimum(df[col], 360 - df[col])

    def _create_basic_features(self, df: pd.DataFrame):
        angle_cols = ["dir", "o"]
        df = self._mirror_angle(df, angle_cols)

        # Height & Weight & BMI
        df["player_height_feet"] = df["player_height"].apply(self._height_to_feet)
        height_parts = df["player_height"].str.split("-", expand=True)
        df["height_inches"] = height_parts[0].astype(float) * 12 + height_parts[
            1
        ].astype(float)
        df["bmi"] = (df["player_weight"] / (df["height_inches"] ** 2)) * 703

        # Velocity & Acceleration & Momentum
        dir_rad = np.deg2rad(df["dir"].fillna(0))
        df["velocity_x"] = df["s"] * np.cos(dir_rad)
        df["velocity_y"] = df["s"] * np.sin(dir_rad)
        df["acceleration_x"] = df["a"] * np.cos(dir_rad)
        df["acceleration_y"] = df["a"] * np.sin(dir_rad)

        df["momentum_x"] = df["velocity_x"] * df["player_weight"]
        df["momentum_y"] = df["velocity_y"] * df["player_weight"]
        df["speed_squared"] = df["s"] ** 2
        df["kinetic_energy"] = 0.5 * df["player_weight"] * df["speed_squared"]

        df["orientation_diff"] = np.abs(df["o"] - df["dir"])
        df["orientation_diff"] = self._warp_angle(df, "orientation_diff")

        df["play_direction"] = (df["play_direction"] == "left").astype(int)

        df["is_offense"] = (df["player_side"] == "Offense").astype(int)
        df["is_defense"] = (df["player_side"] == "Defense").astype(int)
        df["is_receiver"] = (df["player_role"] == "Targeted Receiver").astype(int)
        df["is_coverage"] = (df["player_role"] == "Defensive Coverage").astype(int)
        df["is_passer"] = (df["player_role"] == "Passer").astype(int)

        ball_dx = df["ball_land_x"] - df["x"]
        ball_dy = df["ball_land_y"] - df["y"]
        df["distance_to_ball"] = np.sqrt(ball_dx**2 + ball_dy**2)
        df["angle_to_ball"] = np.arctan2(ball_dy, ball_dx)
        df["ball_direction_x"] = ball_dx / (df["distance_to_ball"] + 1e-6)
        df["ball_direction_y"] = ball_dy / (df["distance_to_ball"] + 1e-6)
        df["angle_diff"] = np.abs(df["o"] - np.degrees(df["angle_to_ball"]))
        df["angle_diff"] = self._warp_angle(df, "angle_diff")
        df["closing_speed"] = (
            df["velocity_x"] * df["ball_direction_x"]
            + df["velocity_y"] * df["ball_direction_y"]
        )

        base = [
            "x",
            "y",
            "s",
            "a",
            "o",
            "dir",
            "frame_id",
            "ball_land_x",
            "ball_land_y",
            "player_weight",
            "player_height_feet",
            "bmi",
            "velocity_x",
            "velocity_y",
            "acceleration_x",
            "acceleration_y",
            "momentum_x",
            "momentum_y",
            "speed_squared",
            "kinetic_energy",
            "orientation_diff",
            "is_offense",
            "is_defense",
            "is_receiver",
            "is_coverage",
            "is_passer",
            "distance_to_ball",
            "angle_to_ball",
            "ball_direction_x",
            "ball_direction_y",
            "angle_diff",
            "closing_speed",
        ]
        self.created_feature_cols.extend([c for c in base if c in df.columns])
        return df

    def _create_target_alignment_features(self, df: pd.DataFrame):
        new_cols = []
        if not {"ball_direction_x", "ball_direction_y"}.issubset(df.columns):
            return df, new_cols

        # Velocity
        if {"velocity_x", "velocity_y"}.issubset(df.columns):
            df["velocity_alignment"] = (
                df["velocity_x"] * df["ball_direction_x"]
                + df["velocity_y"] * df["ball_direction_y"]
            )
            df["velocity_perpendicular"] = (
                df["velocity_x"] * (-df["ball_direction_y"])
                + df["velocity_y"] * df["ball_direction_x"]
            )
            new_cols.extend(["velocity_alignment", "velocity_perpendicular"])

        # Acceleration
        if {"acceleration_x", "acceleration_y"}.issubset(df.columns):
            df["accel_alignment"] = (
                df["acceleration_x"] * df["ball_direction_x"]
                + df["acceleration_y"] * df["ball_direction_y"]
            )
            df["accel_perpendicular"] = (
                df["acceleration_x"] * (-df["ball_direction_y"])
                + df["acceleration_y"] * df["ball_direction_x"]
            )
            new_cols.extend(["accel_alignment", "accel_perpendicular"])

        return df, new_cols

    def _create_multi_window_features(self, df: pd.DataFrame):
        new_cols = []
        mask = df["player_to_predict"]

        df_target = df.loc[mask].copy()

        for window in (3, 5, 10, 20):
            for col in ("velocity_x", "velocity_y", "s"):
                if col in df.columns:
                    r_mean = (
                        df_target.groupby(self.gcols)[col]
                        .rolling(window, min_periods=1)
                        .mean()
                        .reset_index(level=list(range(len(self.gcols))), drop=True)
                    )
                    r_std = (
                        df_target.groupby(self.gcols)[col]
                        .rolling(window, min_periods=1)
                        .std()
                        .reset_index(level=list(range(len(self.gcols))), drop=True)
                    )

                    df.loc[mask, f"{col}_roll{window}"] = r_mean
                    df.loc[mask, f"{col}_std{window}"] = r_std.fillna(0.0)
                    df.loc[mask, f"{col}_dev{window}"] = (
                        df.loc[mask, col].values - r_mean.values
                    )

                    new_cols.extend(
                        [
                            f"{col}_roll{window}",
                            f"{col}_std{window}",
                            f"{col}_dev{window}",
                        ]
                    )

        # speed_trend_ratio
        if "s_roll3" in df.columns and "s_roll20" in df.columns:
            df.loc[mask, "speed_trend_ratio"] = df.loc[mask, "s_roll3"] / (
                df.loc[mask, "s_roll20"] + 1e-3
            )
            new_cols.append("speed_trend_ratio")

        return df, new_cols

    def _create_extended_lag_features(self, df: pd.DataFrame):
        new_cols = []
        mask = df["player_to_predict"]
        df_target = df.loc[mask].copy()

        for lag in (1, 2, 3, 5, 10):
            for col in ("velocity_x", "velocity_y", "s"):
                if col in df.columns:
                    g = df_target.groupby(self.gcols)[col]

                    lagv = g.shift(lag)
                    fillv = lagv.fillna(g.transform("first"))

                    df.loc[mask, f"{col}_lag{lag}"] = fillv[mask]
                    new_cols.append(f"{col}_lag{lag}")

                    if lag <= 3:
                        diffv = df[col] - fillv
                        df.loc[mask, f"{col}_diff_lag{lag}"] = diffv[mask]
                        new_cols.append(f"{col}_diff_lag{lag}")

        return df, new_cols

    def _create_motion_change_features(self, df: pd.DataFrame):
        new_cols = []
        diff_cols = [
            "velocity_x",
            "velocity_y",
            "s",
            "a",
            "dir",
            "o",
        ]

        for col in diff_cols:
            if col not in df.columns:
                continue

            new_col = f"{col}_change"
            df[new_col] = df.groupby(self.gcols)[col].diff().fillna(0.0)
            if col in ["dir", "o"]:
                df[new_col] = wrap_angle_deg(df[new_col])

            new_cols.append(new_col)

        return df, new_cols

    def _create_field_position_features(self, df: pd.DataFrame):
        df["dist_from_left"] = df["y"]
        df["dist_from_right"] = Config.FIELD_Y_MAX - df["y"]
        df["dist_from_sideline"] = np.minimum(
            df["dist_from_left"], df["dist_from_right"]
        )
        df["dist_from_endzone"] = np.minimum(df["x"], Config.FIELD_X_MAX - df["x"])
        df["field_zone_x"] = (df["x"] / Config.FIELD_X_MAX * 5).astype(int).clip(0, 4)
        df["field_zone_y"] = (df["y"] / Config.FIELD_Y_MAX * 3).astype(int).clip(0, 2)
        df["in_red_zone"] = (df["dist_from_endzone"].fillna(np.inf) < 20).astype(np.int8)
        df["near_sideline"] = (df["dist_from_sideline"].fillna(np.inf) < 5).astype(np.int8)
        df["dist_from_center"] = np.hypot(
            df["x"] - Config.FIELD_X_MAX / 2, df["y"] - Config.FIELD_Y_MAX / 2
        )
        return df, [
            "dist_from_sideline",
            "dist_from_endzone",
            "field_zone_x",
            "field_zone_y",
            "in_red_zone",
            "near_sideline",
            "dist_from_center",
        ]

    def _create_distance_rate_features(self, df: pd.DataFrame):
        """Features related to distance to ball"""
        new_cols = []
        if "distance_to_ball" in df.columns:
            d = df.groupby(self.gcols)["distance_to_ball"].diff()
            df["d2ball_dt"] = d.fillna(0.0) * Config.FPS
            df["d2ball_ddt"] = (
                df.groupby(self.gcols)["d2ball_dt"].diff().fillna(0.0) * Config.FPS
            )
            df["time_to_intercept"] = (
                df["distance_to_ball"] / (df["d2ball_dt"].abs() + 1e-3)
            ).clip(0, 10)
            new_cols.extend(["d2ball_dt", "d2ball_ddt", "time_to_intercept"])
        return df, new_cols

    def _create_geometric_features(self, df: pd.DataFrame):
        new_cols = []
        t_total = df["num_frames_output"] / Config.FPS

        df["geo_endpoint_x"] = df["x"] + df["velocity_x"] * t_total
        df["geo_endpoint_y"] = df["y"] + df["velocity_y"] * t_total
        df["geo_endpoint_x"] = df["geo_endpoint_x"].clip(
            Config.FIELD_X_MIN, Config.FIELD_X_MAX
        )
        df["geo_endpoint_y"] = df["geo_endpoint_y"].clip(
            Config.FIELD_Y_MIN, Config.FIELD_Y_MAX
        )
        new_cols.extend(["geo_endpoint_x", "geo_endpoint_y"])

        
        return df, new_cols

    def _create_neighbor_features(self, df: pd.DataFrame):
        new_cols = []
        info_cols = [
            "frame_id",
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "player_side",
            "dir",
            "player_to_predict",
        ]

        info_df = df[self.gcols + info_cols].copy()

        last_df = (
            info_df[info_df["player_to_predict"]]
            .sort_values(self.gcols + ["frame_id"])
            .groupby(self.gcols, as_index=False)
            .tail(1)
            .rename(columns={"frame_id": "frame_id_last"})
            .reset_index(drop=True)
        )

        nb_cols_map = {c: f"{c}_nb" for c in info_cols + ["nfl_id"]}
        info_df = last_df.merge(
            info_df.rename(columns=nb_cols_map),
            left_on=["game_id", "play_id", "frame_id_last"],
            right_on=["game_id", "play_id", "frame_id_nb"],
            how="left",
        )

        info_df.drop(
            columns=["player_to_predict", "player_to_predict_nb"], inplace=True
        )
        info_df = info_df[info_df["nfl_id_nb"] != info_df["nfl_id"]]

        dx = info_df["x_nb"] - info_df["x"]
        dy = info_df["y_nb"] - info_df["y"]
        info_df["dx"] = dx
        info_df["dy"] = dy

        info_df["dvx"] = info_df["velocity_x_nb"] - info_df["velocity_x"]
        info_df["dvy"] = info_df["velocity_y_nb"] - info_df["velocity_y"]

        info_df["dist"] = np.sqrt(info_df["dx"] ** 2 + info_df["dy"] ** 2)
        info_df = info_df[np.isfinite(info_df["dist"]) & (info_df["dist"] > 1e-6)]
        info_df = info_df[info_df["dist"] <= Config.RADIUS]

        info_df["rnk"] = (
            info_df.groupby(self.gcols)["dist"].rank(method="first").astype(int)
        )
        info_df = info_df[info_df["rnk"] <= Config.K_NEIGH]

        info_df["w"] = np.exp(-info_df["dist"] / float(Config.TAU))
        sum_w = info_df.groupby(self.gcols)["w"].transform("sum")
        info_df["wn"] = np.where(sum_w > 0, info_df["w"] / sum_w, 0.0)

        info_df["is_ally"] = (
            info_df["player_side_nb"] == info_df["player_side"]
        ).astype(np.float32)
        info_df["is_opp"] = 1.0 - info_df["is_ally"]

        info_df["wn_ally"] = info_df["wn"] * info_df["is_ally"]
        info_df["wn_opp"] = info_df["wn"] * (1.0 - info_df["is_ally"])

        orig_agg_cols = [
            "momentum_x",
            "momentum_y",
            "kinetic_energy",
            "bmi",
        ]

        diff_agg_cols = [
            "dx",
            "dy",
            "dvx",
            "dvy",
        ]

        for col in orig_agg_cols + diff_agg_cols:
            if col in orig_agg_cols and col not in info_cols:
                continue
            col_nb = f"{col}_nb" if col in orig_agg_cols else col
            info_df[f"{col}_ally_w"] = info_df[col_nb] * info_df["wn_ally"]
            info_df[f"{col}_opp_w"] = info_df[col_nb] * info_df["wn_opp"]

        info_df["dist_ally"] = np.where(
            info_df["is_ally"] > 0.5, info_df["dist"], np.nan
        )
        info_df["dist_opp"] = np.where(
            info_df["is_ally"] < 0.5, info_df["dist"], np.nan
        )

        agg_dict = {}
        for col in orig_agg_cols + diff_agg_cols:
            if col in orig_agg_cols and col not in info_cols:
                continue
            agg_dict[f"{col}_ally_w"] = "sum"
            agg_dict[f"{col}_opp_w"] = "sum"
        agg_dict.update(
            {
                "is_ally": "sum",
                "is_opp": "sum",
                "dist_ally": ["min", "mean"],
                "dist_opp": ["min", "mean"],
            }
        )

        ag = info_df.groupby(self.gcols).agg(agg_dict)
        ag.columns = ["_".join(filter(None, col)).strip() for col in ag.columns.values]
        ag = ag.reset_index()

        ADD_NEAREST_FEAT = True
        if ADD_NEAREST_FEAT:
            K = 3
            near_cols = ["dist"]
            near = info_df.loc[
                info_df["rnk"] <= K, self.gcols + ["rnk"] + near_cols
            ].copy()

            for col in near_cols:
                dwide = near.pivot_table(
                    index=self.gcols, columns="rnk", values=col, aggfunc="first"
                )
                dwide = dwide.rename(
                    columns={i: f"gnn_n{int(i)}_{col}" for i in dwide.columns}
                ).reset_index()
                ag = ag.merge(dwide, on=self.gcols, how="left")

            # px = info_df.groupby(self.gcols)["x"].last().values
            # py = info_df.groupby(self.gcols)["y"].last().values
            # pdir = info_df.groupby(self.gcols)["dir"].last().values

            # for i in range(1, K + 1):
            #     # 邻居坐标与方向
            #     nx = ag[f"gnn_n{i}_x"].values
            #     ny = ag[f"gnn_n{i}_y"].values
            #     ndir = ag[f"gnn_n{i}_dir"].values

            #     # 敌→我方向角
            #     rel_dx = px - nx
            #     rel_dy = py - ny
            #     rel_angle = np.arctan2(rel_dy, rel_dx)

            #     # 敌方速度方向与敌→我方向的夹角（attack_angle）
            #     attack_angle = np.abs(ndir - rel_angle)
            #     attack_angle = np.where(
            #         attack_angle > np.pi, 2 * np.pi - attack_angle, attack_angle
            #     )

            #     # 敌我速度方向差
            #     velocity_angle_diff = np.abs(ndir - pdir)
            #     velocity_angle_diff = np.where(
            #         velocity_angle_diff > np.pi,
            #         2 * np.pi - velocity_angle_diff,
            #         velocity_angle_diff,
            #     )

            #     # 敌我方向一致性指标
            #     attack_alignment = np.cos(attack_angle)
            #     velocity_alignment = np.cos(velocity_angle_diff)

            #     ag[f"gnn_n{i}_attack_angle"] = attack_angle
            #     ag[f"gnn_n{i}_velocity_angle_diff"] = velocity_angle_diff
            #     ag[f"gnn_n{i}_attack_alignment"] = attack_alignment
            #     ag[f"gnn_n{i}_velocity_alignment"] = velocity_alignment

            # ag.drop(
            #     columns=[
            #         f"gnn_n{i}_{col}" for i in range(1, K + 1) for col in near_cols[1:]
            #     ],
            #     inplace=True,
            # )

        # Merge back to df
        new_cols = [c for c in ag.columns if c not in self.gcols]
        for c in new_cols:
            ag[c] = ag[c].fillna(0.0)

        df = df.merge(ag, on=self.gcols, how="left")

        # Defense(Offense) Pressure
        ADD_PRESSURE_FEAT = True
        if ADD_PRESSURE_FEAT:
            df["dist_opp_min"] = df["dist_opp_min"].replace(0, np.nan)
            df["dist_ally_min"] = df["dist_ally_min"].replace(0, np.nan)
            df["dist_opp_eff"] = df["dist_opp_min"].fillna(np.inf)
            df["dist_ally_eff"] = df["dist_ally_min"].fillna(np.inf)

            df["pressure"] = 1 / np.maximum(df["dist_opp_eff"], 0.5)
            df["under_pressure"] = (df["dist_opp_eff"] < 3).astype(int)
            df["have_assistance"] = (
                (df["dist_ally_min"].notna())
                & (df["dist_ally_eff"] < df["dist_opp_eff"])
            ).astype(int)

            df.drop(columns=["dist_opp_eff", "dist_ally_eff"], inplace=True)

            df["pressure_speed"] = df["pressure"] * df["s"]
            df["ally_density"] = df["is_ally_sum"] / (
                np.pi * df["dist_ally_mean"] ** 2 + 1e-6
            )
            df["oppn_density"] = df["is_opp_sum"] / (
                np.pi * df["dist_opp_mean"] ** 2 + 1e-6
            )
            df["density_ratio"] = df["ally_density"] / (df["oppn_density"] + 1e-6)

            new_cols.extend(
                [
                    "pressure",
                    "under_pressure",
                    "have_assistance",
                    "pressure_speed",
                    "ally_density",
                    "oppn_density",
                    "density_ratio",
                ]
            )

        return df, new_cols

    def _create_time_features(self, df: pd.DataFrame):
        new_cols = []

        max_frame = df.groupby(self.gcols)["frame_id"].transform("max")
        df["time_to_end"] = max_frame - df["frame_id"] + df["num_frames_output"]
        df["time_urgency"] = 1 / df["time_to_end"]
        df["time_dist_urgency"] = df["distance_to_ball"] / df["time_to_end"]
        new_cols.extend(["time_to_end", "time_urgency", "time_dist_urgency"])

        df["time_normalized_pass"] = df["frame_id"] / max_frame
        df["time_normalized_all"] = df["frame_id"] / (
            max_frame + df["num_frames_output"]
        )
        new_cols.extend(["time_normalized_pass", "time_normalized_all"])

        return df, new_cols

    def _create_opponent_features(self, df: pd.DataFrame):
        new_cols = []
        return df, new_cols

    def _create_role_features(self, df: pd.DataFrame):
        new_cols = []
        if {"is_receiver", "velocity_alignment"}.issubset(df.columns):
            df["receiver_optimality"] = df["is_receiver"] * df["velocity_alignment"]
            df["receiver_deviation"] = df["is_receiver"] * np.abs(
                df.get("velocity_perpendicular", 0.0)
            )
            df["receiver_speed_usage"] = (
                df["is_receiver"] * df["s"] / (df["s"].max() + 1e-3)
            )
            new_cols.extend(
                ["receiver_optimality", "receiver_deviation", "receiver_speed_usage"]
            )
        if {"is_coverage", "closing_speed"}.issubset(df.columns):
            df["defender_closing_speed"] = df["is_coverage"] * df["closing_speed"]
            df["defender_pressure"] = df["is_coverage"] / (
                df.get("distance_to_ball", 10.0) + 1e-3
            )
            new_cols.extend(["defender_closing_speed", "defender_pressure"])

        return df, new_cols

    def _create_passer_features(self, df: pd.DataFrame):
        # Get (x, y) position of passer
        passer_df = (
            df[df["player_role"] == "Passer"]
            .groupby(["game_id", "play_id", "frame_id"], as_index=False)[["x", "y"]]
            .first()
            .rename(columns={"x": "passer_x", "y": "passer_y"})
        )

        # Merge
        df = df.merge(
            passer_df,
            on=["game_id", "play_id", "frame_id"],
            how="left",
            validate="many_to_one",
        )

        mask = df["player_to_predict"]

        dx = df.loc[mask, "x"].astype("float32") - df.loc[mask, "passer_x"].astype(
            "float32"
        )
        dy = df.loc[mask, "y"].astype("float32") - df.loc[mask, "passer_y"].astype(
            "float32"
        )

        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        ux, uy = dx / dist, dy / dist

        vx = df.loc[mask, "velocity_x"].astype("float32")
        vy = df.loc[mask, "velocity_y"].astype("float32")

        align = vx * ux + vy * uy
        perp = vx * (-uy) + vy * ux

        dir_rad = np.deg2rad(df.loc[mask, "dir"].fillna(0).astype("float32"))

        # bearing
        to_passer_angle = np.arctan2(-dy, -dx)
        bearing = np.rad2deg(to_passer_angle - dir_rad)
        bearing = wrap_angle_deg(bearing)

        pass_dx = df.loc[mask, "ball_land_x"].astype("float32") - df.loc[
            mask, "passer_x"
        ].astype("float32")
        pass_dy = df.loc[mask, "ball_land_y"].astype("float32") - df.loc[
            mask, "passer_y"
        ].astype("float32")
        pass_direction = np.rad2deg(np.arctan2(pass_dy, pass_dx))

        # Drop unused columns
        df.drop(columns=["passer_x", "passer_y"], inplace=True)

        # write back to df
        df.loc[mask, "passer_distance"] = dist
        df.loc[mask, "v_to_passer_alignment"] = align
        df.loc[mask, "v_to_passer_perp"] = perp
        df.loc[mask, "bearing_to_passer"] = bearing
        df.loc[mask, "pass_direction"] = pass_direction

        new_cols = [
            "passer_distance",
            "v_to_passer_alignment",
            "v_to_passer_perp",
            "bearing_to_passer",
            # "pass_direction",
        ]

        return df, new_cols

    def _create_curvature_features(self, df: pd.DataFrame):
        new_cols = []

        dx = df["ball_land_x"] - df["x"]
        dy = df["ball_land_y"] - df["y"]

        a_dir = np.deg2rad(df["dir"].fillna(0.0).values)

        # bearing signed
        bearing = np.arctan2(dy, dx)
        df["bearing_to_land_signed"] = np.rad2deg(
            np.arctan2(np.sin(bearing - a_dir), np.cos(bearing - a_dir))
        )

        # lateral offset (2D cross)
        ux, uy = np.cos(a_dir), np.sin(a_dir)
        df["land_lateral_offset"] = dy * ux - dx * uy

        # curvature
        dir_rad = np.deg2rad(df["dir"].fillna(0.0).values)
        curvature_signed = np.zeros(len(df), dtype="float32")

        df["_grp"] = pd.factorize(df[self.gcols].apply(tuple, axis=1))[0]
        grp_ids, grp_counts = np.unique(df["_grp"], return_counts=True)
        start_idx = 0
        for gid, cnt in tqdm(zip(grp_ids, grp_counts), total=len(grp_ids)):
            idx = slice(start_idx, start_idx + cnt)
            ddir = np.diff(dir_rad[idx], prepend=dir_rad[idx][0])
            # wrap [-pi, pi]
            ddir = (ddir + np.pi) % (2 * np.pi) - np.pi
            # curvature = delta_dir / (s * dt)
            s = df["s"].values[idx].astype("float32")
            curvature_signed[idx] = ddir / (s / Config.FPS + 1e-6)
            start_idx += cnt

        df["curvature_signed"] = curvature_signed
        df["curvature_abs"] = np.abs(curvature_signed)

        # Clear temporary columns
        df.drop(columns="_grp", inplace=True)

        new_cols = [
            "bearing_to_land_signed",
            "land_lateral_offset",
            "curvature_signed",
            "curvature_abs",
        ]

        return df, new_cols

    def _create_route_features(self, df: pd.DataFrame):
        # mask only players to predict
        mask = df["player_to_predict"] == 1
        sub = df[mask].copy()

        # Only use last 5 frames per player
        sub = sub.sort_values(self.gcols + ["frame_id"]).groupby(self.gcols).tail(5)

        # Compute diffs
        sub["dx"] = sub.groupby(self.gcols)["x"].diff()
        sub["dy"] = sub.groupby(self.gcols)["y"].diff()
        sub["ds"] = sub.groupby(self.gcols)["s"].diff()

        # Distance each step
        sub["step_dist"] = np.sqrt(sub["dx"] ** 2 + sub["dy"] ** 2)

        # angles -> second order angle change
        sub["angle"] = np.arctan2(sub["dy"], sub["dx"])
        sub["dangle"] = sub.groupby(self.gcols)["angle"].diff().abs()

        # Total distance & displacement
        feats = (
            sub.groupby(self.gcols)
            .agg(
                traj_total_dist=("step_dist", "sum"),
                start_x=("x", "first"),
                end_x=("x", "last"),
                start_y=("y", "first"),
                end_y=("y", "last"),
                speed_mean=("s", "mean"),
                speed_change=("s", lambda s: s.iloc[-1] - s.iloc[0]),
                traj_turn_ratio=("dangle", lambda a: (a > np.pi / 6).mean()),
            )
            .reset_index()
        )

        # displacement
        feats["traj_dx"] = feats["end_x"] - feats["start_x"]
        feats["traj_dy"] = feats["end_y"] - feats["start_y"]
        feats["traj_displacement"] = np.sqrt(
            feats["traj_dx"] ** 2 + feats["traj_dy"] ** 2
        )

        # straightness
        feats["traj_straightness"] = feats["traj_displacement"] / (
            feats["traj_total_dist"] + 0.1
        )

        # route depth / width / angle
        feats["traj_depth"] = feats["traj_dx"].abs()
        feats["traj_width"] = feats["traj_dy"].abs()
        feats["traj_direction_angle"] = np.arctan2(feats["traj_dy"], feats["traj_dx"])

        # Energy and momentum
        feats["traj_energy"] = feats["speed_mean"] ** 2 * feats["traj_total_dist"]
        feats["traj_momentum"] = feats["speed_mean"] * feats["traj_displacement"]

        turn = (
            sub.groupby(self.gcols)
            .agg(
                traj_max_turn=("dangle", "max"),
                traj_mean_turn=("dangle", "mean"),
            )
            .reset_index()
        )

        # Merge angle features
        feats = feats.merge(turn, on=self.gcols, how="left")

        feat_cols = [
            "speed_mean",
            "speed_change",
            # "traj_turn_ratio",
            "traj_straightness",
            "traj_depth",
            "traj_width",
            "traj_direction_angle",
            # "traj_energy",
            # "traj_momentum",
            "traj_max_turn",
            "traj_mean_turn",
        ]

        # merge back to df (only fill masked rows)
        df = df.merge(
            feats[self.gcols + feat_cols],
            on=self.gcols,
            how="left",
        )

        return df, feat_cols

    def _create_cooperation_features(self, df: pd.DataFrame, K: int = 3):
        new_cols = []

        info_cols = [
            "frame_id",
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "s",
            "a",
            "player_side",
        ]

        # --- Step 1: 提取每个球员在最后一帧的状态 ---
        info_df = df[self.gcols + info_cols].copy()
        last_df = (
            info_df.sort_values(self.gcols + ["frame_id"])
            .groupby(self.gcols, as_index=False)
            .tail(1)
            .rename(columns={"frame_id": "frame_id_last"})
            .reset_index(drop=True)
        )

        nb_cols_map = {c: f"{c}_nb" for c in info_cols}

        # 与同一帧的其他球员（同队）配对
        info_df = last_df.merge(
            info_df.rename(columns=nb_cols_map),
            left_on=["game_id", "play_id", "frame_id_last"],
            right_on=["game_id", "play_id", "frame_id_nb"],
            how="left",
        )

        info_df = info_df[
            (info_df["nfl_id_nb"] != info_df["nfl_id"])
            & (info_df["player_side_nb"] == info_df["player_side"])
        ]

        # --- Step 2: 计算相对空间与运动差 ---
        dx = info_df["x_nb"] - info_df["x"]
        dy = info_df["y_nb"] - info_df["y"]
        info_df["dist"] = np.sqrt(dx**2 + dy**2)

        # 邻居距离筛选
        info_df = info_df[np.isfinite(info_df["dist"]) & (info_df["dist"] > 1e-3)]
        info_df = info_df[info_df["dist"] <= getattr(Config, "RADIUS", 15.0)]

        # 最近队友（rank=1）
        info_df["rnk"] = info_df.groupby(self.gcols)["dist"].rank(method="first")
        info_df = info_df[info_df["rnk"] <= K]

        # --- Step 3: 协同特征计算 ---
        info_df["vx_diff"] = info_df["velocity_x_nb"] - info_df["velocity_x"]
        info_df["vy_diff"] = info_df["velocity_y_nb"] - info_df["velocity_y"]
        info_df["speed_diff"] = info_df["s_nb"] - info_df["s"]
        info_df["acc_diff"] = info_df["a_nb"] - info_df["a"]

        # 角度差：方向是否一致
        info_df["angle_self"] = np.arctan2(info_df["velocity_y"], info_df["velocity_x"])
        info_df["angle_nb"] = np.arctan2(
            info_df["velocity_y_nb"], info_df["velocity_x_nb"]
        )
        info_df["angle_diff"] = np.abs(info_df["angle_self"] - info_df["angle_nb"])
        info_df["angle_diff"] = np.where(
            info_df["angle_diff"] > np.pi,
            2 * np.pi - info_df["angle_diff"],
            info_df["angle_diff"],
        )
        info_df["heading_align"] = np.cos(info_df["angle_diff"])  # 方向一致性 [−1,1]

        # 是否在靠近 / 远离
        info_df["approaching_rate"] = (
            dx * info_df["vx_diff"] + dy * info_df["vy_diff"]
        ) / (info_df["dist"] + 1e-3)

        # --- Step 4: 聚合（以每个玩家为中心） ---
        ag = (
            info_df.groupby(self.gcols)
            .agg(
                coop_nearest_dist=("dist", "min"),
                coop_speed_diff=("speed_diff", "mean"),
                coop_acc_diff=("acc_diff", "mean"),
                coop_heading_align=("heading_align", "mean"),
                coop_approaching=("approaching_rate", "mean"),
            )
            .reset_index()
        )

        # --- Step 5: 合并到主 df ---
        new_cols = [
            "coop_nearest_dist",
            "coop_speed_diff",
            "coop_acc_diff",
            "coop_heading_align",
            "coop_approaching",
        ]
        for c in new_cols:
            ag[c] = ag[c].fillna(0.0)

        df = df.merge(ag[self.gcols + new_cols], on=self.gcols, how="left")
        return df, new_cols

    def _create_receiver_features(self, df: pd.DataFrame):
        """Almost the same as `self._create_passer_features`"""
        # Get (x, y) position of receiver
        receiver_df = (
            df[df["player_role"] == "Targeted Receiver"]
            .groupby(["game_id", "play_id", "frame_id"], as_index=False)[["x", "y"]]
            .first()
            .rename(columns={"x": "receiver_x", "y": "receiver_y"})
        )

        df = df.merge(
            receiver_df,
            on=["game_id", "play_id", "frame_id"],
            how="left",
            validate="many_to_one",
        )
        mask = df["player_to_predict"]

        dx = df.loc[mask, "x"].astype("float32") - df.loc[mask, "receiver_x"].astype(
            "float32"
        )
        dy = df.loc[mask, "y"].astype("float32") - df.loc[mask, "receiver_y"].astype(
            "float32"
        )

        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        ux, uy = dx / dist, dy / dist

        vx = df.loc[mask, "velocity_x"].astype("float32")
        vy = df.loc[mask, "velocity_y"].astype("float32")

        # Projection
        align = vx * ux + vy * uy
        perp = vx * (-uy) + vy * ux

        dir_rad = np.deg2rad(df.loc[mask, "dir"].fillna(0).astype("float32"))

        # bearing
        to_receiver_angle = np.arctan2(-dy, -dx)
        bearing = np.rad2deg(to_receiver_angle - dir_rad)
        bearing = wrap_angle_deg(bearing)

        # write back to df
        df.loc[mask, "receiver_distance"] = dist
        df.loc[mask, "v_to_receiver_alignment"] = align
        df.loc[mask, "v_to_receiver_perp"] = perp
        df.loc[mask, "bearing_to_receiver"] = bearing

        new_cols = [
            "receiver_distance",
            "v_to_receiver_alignment",
            "v_to_receiver_perp",
            "bearing_to_receiver",
        ]

        return df, new_cols

    def transform(self, df: pd.DataFrame):
        df = df.copy().sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
        # # Use index to accelerate groupby and merge operations
        # df.set_index(self.gcols, inplace=True, drop=False)
        df = self._create_basic_features(df)

        # TODO: Optimize for interactive=False
        for group_name in self.active_groups:
            if group_name in self.feature_creators:
                creator, interactive = self.feature_creators[group_name]
                start_time = time.time()
                df, new_cols = creator(df)
                elapsed = time.time() - start_time
                self.created_feature_cols.extend(new_cols)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] [+] Added '{group_name}' "
                    f"({len(new_cols)} cols) in {elapsed:.2f}s\n    Columns: {new_cols}"
                )
            else:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] [!] Unknown feature group: {group_name}"
                )

        df = df[df["player_to_predict"]]
        final_cols = sorted(set(self.created_feature_cols))
        print(f"\nTotal features created: {len(final_cols)}")
        return df, final_cols