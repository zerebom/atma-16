from pathlib import Path
from typing import Literal

import polars as pl
from scorta.feature.feature import Feature

from atma_16.dataset.dataset import AtmaData16Loader
from atma_16.utils.polars_utils import over_rank


class AtmaFeature(Feature):
    def __init__(
        self,
        output_dir: str | Path,
        data_loader: AtmaData16Loader,
        feature_cols: list[str],
        user_col: str = "session_id",
        item_col: str = "yado_no",
        key_cols: list[str] = ["session_id", "yado_no"],
        suffix: str | None = None,
        mode: Literal["train", "test"] | None = None,
    ):
        super().__init__(
            output_dir=output_dir,
            feature_cols=feature_cols,
            user_col=user_col,
            item_col=item_col,
            key_cols=key_cols,
            suffix=suffix,
            mode=mode,
        )
        self.data_loader = data_loader


class SessionYadoRawFeature(AtmaFeature):
    def fit(self, df: pl.DataFrame):
        out_df = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()
        return out_df[self.key_cols + self.feature_cols]


class YadoSesMeanFeature(AtmaFeature):
    """
    yadごとのsessionの平均値
    どれくらいの長いsessionに含まれてるか.train/testどちらに多くでるか
    """

    def fit(self, df: pl.DataFrame):
        def agg_and_attach_feat_log(log):
            return (
                log.with_columns((pl.col("seq_no").max().over("session_id") + 1).alias("session_seq_len"))
                .group_by("yad_no")
                .agg(pl.count(), pl.mean("session_seq_len"))
            )

        train_log = self.data_loader.load_train_log()
        test_log = self.data_loader.load_test_log()

        A, B = ["this", "another"] if self.mode == "train" else ["another", "this"]

        out_df = (
            agg_and_attach_feat_log(train_log)
            .join(agg_and_attach_feat_log(test_log), on="yad_no", how="outer")
            .fill_null(0)
            .select(
                pl.col("yad_no"),
                pl.col("session_seq_len").alias(f"{A}_session_seq_len"),
                pl.col("session_seq_len_right").alias(f"{B}_session_seq_len"),
                pl.col("count").alias(f"{A}_count"),
                pl.col("count_right").alias(f"{B}_count"),
                (pl.col("count") / (pl.col("count") + pl.col("count_right"))).alias(f"{A}_ratio"),
                (pl.col("count_right") / (pl.col("count") + pl.col("count_right"))).alias(f"{B}_ratio"),
            )
        )

        out_df = out_df.select(pl.col("yad_no"), pl.col("*").exclude("yad_no").name.suffix("_yad_mean"))
        return out_df[self.key_cols + self.feature_cols]


class YadoRawFeature(AtmaFeature):
    def fit(self, df: pl.DataFrame):
        out_df = self.data_loader.load_yado()
        # cd_cols = ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
        # out_df = out_df.join(yado_feat, how="left", on=self.key_cols)
        # out_df = out_df.with_columns([pl.col(c).cast(pl.Utf8).cast(pl.Categorical) for c in cd_cols])

        # print(out_df.shape)
        return out_df[self.key_cols + self.feature_cols]


class YadoRankFeature(AtmaFeature):
    """
    cd内でのyadの人気順位
    """

    def fit(self, df: pl.DataFrame):
        yad = self.data_loader.load_yado()
        train_log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()
        agg_df = train_log.group_by("yad_no").agg(pl.count().alias("yad_cnt"))

        cd_cols = ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
        agg_df = agg_df.join(yad[["yad_no"] + cd_cols], how="left", on="yad_no")
        agg_df = agg_df.with_columns(
            [pl.col("yad_no").rank(method="ordinal", descending=True).alias("yad_cnt_rank")]
            + [over_rank("yad_cnt", c).alias(f"yad_cnt_rank_{c}") for c in cd_cols]
        ).sort(["ken_cd", "yad_cnt_rank_ken_cd"])
        print(agg_df.shape)
        return agg_df[self.key_cols + self.feature_cols]


class SessionFeature(AtmaFeature):
    """session内のyadの集約値"""

    def fit(self, df: pl.DataFrame):
        yado = self.data_loader.load_yado()
        log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()

        mode_cols = ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
        mean_cols = [
            "total_room_cnt",
            "wireless_lan_flg",
            "onsen_flg",
            "kd_stn_5min",
            "kd_bch_5min",
            "kd_slp_5min",
            "kd_conv_walk_5min",
        ]

        agg_df = (
            log.join(yado, how="left", on="yad_no")
            .group_by("session_id")
            .agg(
                [pl.col(c).mode().alias(f"user_mode_{c}") for c in mode_cols]
                + [pl.count("seq_no").alias("user_seq_cnt")]
                + [pl.col(c).mean().alias(f"user_mean_{c}") for c in mean_cols]
                + [pl.col(c).std().alias(f"user_std_{c}") for c in mean_cols]
            )
            .with_columns([pl.col(f"user_mode_{c}").list.get(0) for c in mode_cols])
        )
        return agg_df[self.key_cols + self.feature_cols]


class MatchModeCDFeature(AtmaFeature):
    def fit(self, df: pl.DataFrame):
        assert "user_mode_wid_cd" in df.columns, df.columns
        out_df = df.with_columns(
            [
                (pl.col(f"user_mode_{col}") == pl.col(col)).cast(int).alias(f"is_match_mode_{col}")
                for col in ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
            ]
        )
        return out_df[self.key_cols + self.feature_cols]


class DiffUserMeanAndPredFeature(AtmaFeature):
    """ユーザー平均値と予測対象のdiff"""

    def fit(self, df: pl.DataFrame):
        assert "user_mean_total_room_cnt" in df.columns

        out_df = df.with_columns(
            [(pl.col(f"user_mean_{col}") == pl.col(col)).alias(f"diff_{col}") for col in ["total_room_cnt"]]
        )
        return out_df[self.key_cols + self.feature_cols]


class DiffUserLastAndPredFeature(AtmaFeature):
    def __init__(
        self,
        output_dir: str | Path,
        data_loader: AtmaData16Loader,
        feature_cols: list[str],
        yad_ses_feat: YadoSesMeanFeature,
        user_col: str = "session_id",
        item_col: str = "yado_no",
        key_cols: list[str] = ["session_id", "yado_no"],
        suffix: str | None = None,
        mode: Literal["train", "test"] | None = None,
    ):
        super().__init__(output_dir, data_loader, feature_cols, user_col, item_col, key_cols, suffix, mode)
        self.feat = yad_ses_feat.load()

    def fit(self, df: pl.DataFrame):
        cols = [
            "this_session_seq_len_yad_mean",
            "another_session_seq_len_yad_mean",
            "this_count_yad_mean",
            "another_count_yad_mean",
        ]
        log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()

        last_log = log.group_by("session_id").tail(1)

        last_feat = self.feat.select(
            [pl.col("yad_no")] + [pl.col(col).alias(f"{col}_last").cast(pl.Float32) for col in cols]
        )
        pred_feat = self.feat.select(
            [pl.col("yad_no")] + [pl.col(col).alias(f"{col}_pred").cast(pl.Float32) for col in cols]
        )

        diff_yad_df = (
            df.join(last_log.join(last_feat, how="left", on="yad_no"), how="left", on="session_id")
            .join(pred_feat, how="left", on="yad_no")
            .select(
                [pl.col("session_id", "yad_no")]
                + [(pl.col(f"{col}_last") - pl.col(f"{col}_pred")).alias(f"diff_{col}") for col in cols]
            )
        )
        return diff_yad_df


class SeenedYadoFeature(AtmaFeature):
    def fit(self, df: pl.DataFrame):
        all_log = self.data_loader.load_all_log()
        seened_df = all_log.pivot(index="session_id", columns="seq_no", values="yad_no").select(
            pl.col("session_id"), pl.col("*").exclude("session_id").name.prefix("seened_yad_no_")
        )
        return seened_df[self.key_cols + self.feature_cols]


class SmlCDStatsFeatre(AtmaFeature):
    def fit(self, df: pl.DataFrame):
        all_log = self.data_loader.load_all_log()
        yado = self.data_loader.load_yado()

        cols = [
            "total_room_cnt",
            "wireless_lan_flg",
            "onsen_flg",
            "kd_stn_5min",
            "kd_bch_5min",
            "kd_slp_5min",
            "kd_conv_walk_5min",
        ]
        sml_cd_df = (
            all_log.join(yado.fill_null(0), on="yad_no", how="left")
            .group_by("sml_cd")
            .agg(
                [
                    pl.col("yad_no").n_unique().alias("sml_cd_yad_nunique"),
                    pl.col("yad_no").count().alias("sml_cd_yad_cnt"),
                ]
                + [pl.col(c).mean().alias(f"sml_cd_mean_{c}") for c in cols]
                + [pl.col(c).std().alias(f"sml_cd_std_{c}") for c in cols],
            )
        )
        return sml_cd_df[self.key_cols + self.feature_cols]
