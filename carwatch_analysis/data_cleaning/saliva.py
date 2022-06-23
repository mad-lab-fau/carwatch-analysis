from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.utils.datatype_helper import SalivaRawDataFrame, is_saliva_raw_dataframe

__all__ = ["clean_missing_values"]


def clean_missing_values(data: SalivaRawDataFrame, print_output: Optional[bool] = True) -> SalivaRawDataFrame:
    is_saliva_raw_dataframe(data, "cortisol")

    missing_mask = (data["cortisol"].unstack("sample").isna()).any(axis=1)
    missing_mask = np.logical_or((data["cortisol"].unstack("sample") < 0.75).any(axis=1), missing_mask)

    data_out = data.loc[~missing_mask, :]
    if print_output:
        print(
            f"Missing Values\n"
            f"Before: {data.unstack('sample').shape[0]} | After: {data_out.unstack('sample').shape[0]}"
        )
    return data_out


def clean_missing_date_information(data: SalivaRawDataFrame, print_output: Optional[bool] = True) -> SalivaRawDataFrame:
    is_saliva_raw_dataframe(data, "cortisol")

    data_out = data.dropna(subset=["date", "time_abs", "wake_onset_time"])

    if print_output:
        print(
            "Missing Date Information\n"
            f"Before: {data.unstack('sample').shape[0]} | After: {data_out.unstack('sample').shape[0]}"
        )
    return data_out


def clean_s0_after_wake_onset(data: SalivaRawDataFrame, print_output: Optional[bool] = True) -> SalivaRawDataFrame:
    is_saliva_raw_dataframe(data, "cortisol")

    times = data.xs("S0", level="sample")[["wake_onset_time", "time_abs"]]
    wo_mask = np.abs(times.diff(axis=1)["time_abs"]) > pd.Timedelta("5min")

    data_out = data.loc[~wo_mask, :]

    if print_output:
        print(
            "Difference between S0 and Wake Onset > 5 min\n"
            f"Before: {data.unstack('sample').shape[0]} | After: {data_out.unstack('sample').shape[0]}"
        )
    return data_out


def clean_sampling_time_difference(data: SalivaRawDataFrame, print_output: Optional[bool] = True) -> SalivaRawDataFrame:
    is_saliva_raw_dataframe(data, "cortisol")

    time_mask = ((data["time"].unstack(level="sample").diff(axis=1) - 15).abs() > 5).any(axis=1)
    data_out = data.loc[~time_mask]

    if print_output:
        print(
            "Difference between consecutive samples > 5 min from actual time interval\n"
            f"Before: {data.unstack('sample').shape[0]} | After: {data_out.unstack('sample').shape[0]}"
        )
    return data_out


def clean_statistical_outlier(data: SalivaRawDataFrame, print_output: Optional[bool] = True) -> SalivaRawDataFrame:
    """more than 3 std from mean"""
    is_saliva_raw_dataframe(data, "cortisol")

    outlier_mask = (
        data["cortisol"].unstack("sample").transform(lambda df: (df - df.mean()) / df.std()).abs() > 3.0
    ).any(axis=1)
    data_out = data.loc[~outlier_mask]

    if print_output:
        print(
            "Statistical Outlier\n"
            f"Before: {data.unstack('sample').shape[0]} | After: {data_out.unstack('sample').shape[0]}"
        )
    return data_out


def clean_physiological_outlier(data: SalivaRawDataFrame, print_output: Optional[bool] = True) -> SalivaRawDataFrame:
    """More than 70 nmol/l"""

    phys_mask = (data["cortisol"].unstack("sample") > 70).any(axis=1)
    data_out = data.loc[~phys_mask]

    if print_output:
        print(
            "Physiological Outlier\n"
            f"Before: {data.unstack('sample').shape[0]} | After: {data_out.unstack('sample').shape[0]}"
        )
    return data_out
