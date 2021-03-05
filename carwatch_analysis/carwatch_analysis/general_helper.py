from typing import Sequence

import pandas as pd

from biopsykit.utils import path_t


def subject_id_from_path(path: path_t) -> str:
    return str(path.name).split("_")[0]


def concat_nights(list_df: Sequence[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(list_df, ignore_index=True)
    df.index.name = "night"
    return df
