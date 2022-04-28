from typing import Optional

import pandas as pd


def clean_statistical_outlier(data: pd.DataFrame, print_output: Optional[bool] = True) -> pd.DataFrame:
    """more than 3 std from mean"""

    outlier_mask = (
        data["data"].unstack("imu_feature")
        # .groupby(["wakeup_type", "time_span"])
        .transform(lambda df: (df - df.mean()) / df.std()).abs()
        > 3.0
    ).any(axis=1)

    if print_output:
        print("Statistical Outlier\nRemoved the following outlier:")
        print(outlier_mask.sum())
        # print(outlier_mask.groupby(["wakeup_type", "time_span"]).sum())

    data_out = data.loc[~outlier_mask]
    data_out = data_out.unstack("imu_feature").dropna().stack()
    return data_out
