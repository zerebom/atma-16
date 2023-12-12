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


class YadoRawFeature(AtmaFeature):
    def fit(self, df: pl.DataFrame):
        out_df = self.data_loader.load_yado()
        # cd_cols = ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
        # out_df = out_df.join(yado_feat, how="left", on=self.key_cols)
        # out_df = out_df.with_columns([pl.col(c).cast(pl.Utf8).cast(pl.Categorical) for c in cd_cols])

        # print(out_df.shape)
        return out_df[self.key_cols + self.feature_cols]


class YadoRankFeature(AtmaFeature):
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
            )
            .with_columns([pl.col(f"user_mode_{c}").list.get(0) for c in mode_cols])
        )
        return agg_df[self.key_cols + self.feature_cols]
