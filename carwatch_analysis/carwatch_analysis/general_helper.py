from typing import Sequence, Optional

import pandas as pd
from carwatch_analysis._types import path_t


def subject_id_from_path(path: path_t) -> str:
    return str(path.name).split("_")[0]


def concat_nights(list_df: Sequence[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if len(list_df) == 0:
        return None
    df = pd.concat(list_df, ignore_index=True)
    df.index.name = "night"
    return df


def describe_groups_df(data: pd.DataFrame, variable: str, order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    data = data.unstack().mean(axis=1).groupby(variable).describe()
    if order:
        data = data.reindex(order)
    return data
