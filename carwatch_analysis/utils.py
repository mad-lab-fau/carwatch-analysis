from typing import Optional, Sequence

import pandas as pd


def describe_groups_df(data: pd.DataFrame, variable: str, order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    data = data.unstack().mean(axis=1).groupby(variable).describe()
    if order:
        data = data.reindex(order)
    return data
