from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold
from tqdm import tqdm

DataFrame: TypeAlias = pl.DataFrame | pd.DataFrame
DfType: TypeAlias = Literal["pl", "pd"]


class AtmaData16Loader:
    def __init__(self, input_dir: str | Path):
        self.input_dir = input_dir
        self.csv_paths = input_dir.glob("*.csv")
        self.idx2ses, self.ses2idx = self.load_ses2idx()

    def csv2parquet(self):
        for csv_path in tqdm(self.csv_paths):
            parquet_path = self.input_dir / Path(csv_path.stem + ".parquet")
            pl.read_csv(csv_path).write_parquet(parquet_path)

    @staticmethod
    def _load_parquet(path: Path, frame_type: DfType) -> DataFrame:
        return pl.read_parquet(path) if frame_type == "pl" else pd.read_parquet(path)

    def load_test_log(self, frame_type: DfType = "pl", convert: bool = True) -> DataFrame:
        df = self._load_parquet(self.input_dir / "test_log.parquet", frame_type)
        return self.convert_ses2idx(df) if convert else df

    def load_train_log(self, frame_type: DfType = "pl", convert: bool = True) -> DataFrame:
        df = self._load_parquet(self.input_dir / "train_log.parquet", frame_type)
        return self.convert_ses2idx(df) if convert else df

    def load_all_log(self, frame_type: DfType = "pl", convert: bool = True) -> DataFrame:
        df = pl.concat([self.load_train_log("pl", convert=False), self.load_test_log("pl", convert=False)])
        df = self.convert_ses2idx(df) if convert else df
        return df if frame_type == "pl" else df.to_pandas()

    def load_ses2idx(self) -> tuple[dict[int, str], dict[str, int]]:
        idx2ses = dict(enumerate(self.load_all_log("pd", convert=False)["session_id"].unique()))
        ses2idx = {k: idx for idx, k in idx2ses.items()}
        assert ses2idx["000007603d533d30453cc45d0f3d119f"] == 0
        assert idx2ses[0] == "000007603d533d30453cc45d0f3d119f"
        return idx2ses, ses2idx

    def load_train_label(
        self, frame_type: DfType = "pl", assign_fold: bool = True, fold_num: int = 5, seed: int = 113
    ) -> DataFrame:
        label = self._load_parquet(self.input_dir / "train_label.parquet", frame_type)
        if assign_fold:
            kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
            fold_assignments = np.full(label.height, -1, dtype=int)

            for i, (_, valid_index) in enumerate(kf.split(label)):
                fold_assignments[valid_index] = i
            label = label.with_columns(pl.Series("fold", fold_assignments))

        return self.convert_ses2idx(label)

    def load_yado(self, frame_type: DfType = "pl") -> DataFrame:
        yado = self._load_parquet(self.input_dir / "yado.parquet", "pl")
        cd_cols = ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
        for col in cd_cols:
            idx2cat = dict(enumerate(yado[col].unique(maintain_order=True)))
            cat2idx = {k: idx for idx, k in idx2cat.items()}
            yado = yado.with_columns(pl.col(col).replace(cat2idx, default=None)).fill_null(0)

        return yado if frame_type == "pl" else yado.to_pandas()

    def load_all_dfs(self, frame_type: DfType = "pl") -> dict[str, DataFrame]:
        return {path.stem: self._load_parquet(path, frame_type) for path in self.input_dir.glob("*.parquet")}

    def load_sample_submission(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "sample_submission.parquet", frame_type)

    def convert_ses2idx(self, df: pl.DataFrame, col: str = "session_id") -> pl.DataFrame:
        return df.with_columns(pl.col(col).replace(self.ses2idx, default=None))

    def convert_idx2ses(self, df: pl.DataFrame, col: str = "session_id") -> pl.DataFrame:
        return df.with_columns(pl.col(col).replace(self.idx2ses, default=None))
