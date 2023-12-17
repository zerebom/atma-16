from typing import Any, Iterator

import pandas as pd
import polars as pl

from atma_16.dataset.dataset import AtmaData16Loader


class ColumnBasedCV:
    def __init__(self, dl: AtmaData16Loader):
        self.label = dl.load_train_label()

    def split(self, X: pl.DataFrame, y: Any, **kwargs: Any) -> Iterator[tuple[np.array, np.array]]:
        for fold in range(5):
            idx_pair = (
                pl.DataFrame(X).join(self.label, on=["session_id"], how="left").with_row_count()[["row_nr", "fold"]]
            )

            train_idx = idx_pair.filter(pl.col("fold") != fold)["row_nr"].to_numpy()
            test_idx = idx_pair.filter(pl.col("fold") == fold)["row_nr"].to_numpy()
            yield train_idx, test_idx
