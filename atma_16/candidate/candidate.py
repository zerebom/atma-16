from pathlib import Path
from typing import Iterator, Literal

import polars as pl
from scorta.recsys.candidate_generate import Candidate

from atma_16.dataset.dataset import AtmaData16Loader
from atma_16.utils.polars_utils import min_max_scaler, over_rank


class Atma16Candidate(Candidate):
    def __init__(
        self,
        data_loader: AtmaData16Loader,
        output_dir: str | Path,
        mode: Literal["train", "test"] = "train",
        user_col: str | None = "session_id",
        item_col: str | None = "yad_no",
        target_df: pl.DataFrame | None = None,
        top_k: int = 10,
        fold_num: int = 5,
    ):
        super().__init__(output_dir, user_col, item_col, target_df=target_df, suffix=mode)
        self.mode = mode
        self.data_loader = data_loader
        self.top_k = top_k
        self.fold_num = fold_num


class LastSeenedCandidate(Atma16Candidate):
    """同一session内の宿. 最後に見た宿ほどスコアが高くなるようにする"""

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


class PopularAtCDCandidate(Atma16Candidate):
    """同一cd内の宿. おおくのsessionで見られている宿ほどスコアが高くなるようにする"""

    def generate(self) -> pl.DataFrame:
        yado = self.data_loader.load_yado()
        all_log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()

        yad_pop = (
            all_log.join(yado, on="yad_no", how="left")
            .group_by(["yad_no", "sml_cd"])
            .agg(pl.count())
            .filter(over_rank("count", "yad_no") <= self.top_k)
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
            .filter(pl.col("rank") < self.top_k)
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


class CoVisitCandidate(Atma16Candidate):
    def generate(self) -> pl.DataFrame:
        all_log = self.data_loader.load_all_log()
        log = self.data_loader.load_train_log() if self.mode == "train" else self.data_loader.load_test_log()
        item2item_df = (
            all_log.join(all_log, on="session_id")
            .filter(pl.col("yad_no") != pl.col("yad_no_right"))
            .group_by("yad_no", "yad_no_right")
            .count()
        )

        co_view_df = (
            log.join(item2item_df, left_on=["yad_no"], right_on=["yad_no_right"], how="left")
            .group_by(["session_id", "yad_no_right"])
            .agg(pl.col("count").sum().alias("count"))
            .rename({"yad_no_right": "yad_no"})
            .select(
                pl.col(["session_id", "yad_no"]),
                over_rank("count", "session_id").alias("rank"),
                min_max_scaler("count").alias("score"),
            )
            .sort("session_id", "rank")
        )
        return co_view_df


class TopBookedFromLastViewCandidate(Atma16Candidate):
    def generate(self) -> pl.DataFrame:
        label = self.data_loader.load_train_label()
        train_log = (
            self.data_loader.load_train_log().join(label, how="left", on="session_id").rename({"yad_no_right": "label"})
        )
        test_log = self.data_loader.load_test_log()

        out_df = pl.DataFrame()
        for train, test in self.generate_train_test(train_log, test_log):
            tmp_df = self.calc_top_booked_hotel_from_last_view(train, test)
            out_df = pl.concat([out_df, tmp_df])
        return out_df

    def generate_train_test(
        self, train_log: pl.DataFrame, test_log: pl.DataFrame
    ) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        if self.mode == "train":
            for i in range(self.fold_num):
                train = train_log.filter(pl.col("fold") != i).drop("fold")
                test = train_log.filter(pl.col("fold") == i).drop(["fold", "label"])
                yield train, test
        else:
            yield train_log.drop("fold"), test_log

    def calc_top_booked_hotel_from_last_view(self, train_log: pl.DataFrame, test_log: pl.DataFrame) -> pl.DataFrame:
        latest_log = train_log.group_by("session_id").tail(1).rename({"yad_no": "latest_yad_no"})
        co_yads_w_latest = (
            latest_log.group_by(["latest_yad_no", "label"])
            .count()
            .sort(by=["latest_yad_no", "count"], descending=[False, True])
            .with_columns(over_rank("count", "latest_yad_no").alias("rank"))
            .filter(pl.col("rank") <= 10)
        )

        out_df = (
            test_log.group_by("session_id")
            .tail(1)
            .join(co_yads_w_latest, how="left", left_on="yad_no", right_on="latest_yad_no")
            .select(
                pl.col("session_id"),
                pl.col("label").alias("yad_no"),
                pl.col("rank"),
                min_max_scaler("count").alias("score"),
            )
        )
        return out_df
