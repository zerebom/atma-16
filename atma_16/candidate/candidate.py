from pathlib import Path
from typing import Literal

import polars as pl
from scorta.recsys.candidate_generate import Candidate

from atma_16.dataset.dataset import AtmaData16Loader
from atma_16.utils.polars_utils import min_max_scaler, over_rank


class LastSeenedCandidate(Candidate):
    def __init__(
        self,
        data_loader: AtmaData16Loader,
        output_dir: str | Path,
        mode: Literal["train", "test"] = "train",
        user_col: str | None = "session_id",
        item_col: str | None = "yad_no",
        target_df: pl.DataFrame | None = None,
    ):
        """同一session内の宿. 最後に見た宿ほどスコアが高くなるようにする"""
        super().__init__(output_dir, user_col, item_col, target_df=target_df, suffix=mode)
        self.mode = mode
        self.data_loader = data_loader

    def generate(self) -> pl.DataFrame:
        log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()
        seened_df = (
            log.with_columns(
                (pl.col("seq_no").max().over("session_id") == pl.col("seq_no")).alias("is_last"),
            )
            .filter(~pl.col("is_last"))
            .with_columns(
                pl.col("seq_no").max().over("session_id") - pl.col("seq_no") + pl.lit(1)
            )  # 最大値からの差分 + 1
            .with_columns(min_max_scaler("seq_no").over("session_id").alias("score"))
            .sort("session_id", "seq_no")
            .with_columns(over_rank("score", "session_id").alias("rank"))
            .fill_nan(1)  # 最も卑近なら1,だんだん0に近づく
            .drop(["is_last", "seq_no"])
        )
        return seened_df


class PopularAtCDCandidate(Candidate):
    def __init__(
        self,
        data_loader: AtmaData16Loader,
        output_dir: str | Path,
        mode: Literal["train", "test"] = "train",
        user_col: str | None = "session_id",
        item_col: str | None = "yad_no",
        target_df: pl.DataFrame | None = None,
        topk=10,
    ):
        """同一cd内の宿. おおくのsessionで見られている宿ほどスコアが高くなるようにする"""
        super().__init__(output_dir, user_col, item_col, target_df=target_df, suffix=mode)
        self.mode = mode
        self.data_loader = data_loader
        self.topk = topk

    def generate(self) -> pl.DataFrame:
        yado = self.data_loader.load_yado()
        all_log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()

        yad_pop = (
            all_log.join(yado, on="yad_no", how="left")
            .group_by(["yad_no", "sml_cd"])
            .agg(pl.count())
            .filter(over_rank("count", "yad_no") <= self.topk)
            .sort(by=["sml_cd", "count"], descending=True)
        )

        agg_logs = (
            all_log.join(yado, on="yad_no", how="left")
            .join(yad_pop, on="sml_cd")
            .group_by(["session_id", "yad_no_right"])
            .agg(pl.sum("count").alias("sum"))
        )

        out_df = (
            agg_logs.with_columns(over_rank("sum", "session_id").alias("rank"))
            .sort(by=["session_id", "rank"])
            .filter(pl.col("rank") < self.topk)
            .select(
                [
                    pl.col("session_id"),
                    pl.col("yad_no_right").alias("yad_no"),
                    min_max_scaler("sum").alias("score"),
                    pl.col("rank"),
                ]
            )
        )
        return out_df
