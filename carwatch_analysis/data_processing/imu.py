import json
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
from biopsykit.io import load_long_format_csv
from biopsykit.signals.imu.static_moment_detection import find_static_moments
from biopsykit.signals.imu.feature_extraction.static_moments import compute_features
from biopsykit.sleep.sleep_endpoints import endpoints_as_df
from biopsykit.sleep.sleep_processing_pipeline import predict_pipeline_acceleration

from carwatch_analysis._types import path_t
from carwatch_analysis.datasets import CarWatchDatasetRaw
from carwatch_analysis.exceptions import (
    ImuDataNotFoundException,
    ImuInvalidDateException,
    NoSuitableImuDataFoundException,
    DateNotAvailableException,
)


def process_night(
    subset: CarWatchDatasetRaw, compute_endpoints: bool, compute_features: bool, **kwargs
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not subset.subject_folder_path.exists():
        return None, None
    export_path = subset.subject_folder_path.joinpath("processed")
    export_path.mkdir(exist_ok=True)

    endpoints_selfreport = subset.endpoints_selfreport

    subject_id = subset.index["subject"][0]
    night_id = subset.index["night"][0]

    # check whether old processing results already exist
    feature_file = export_path.joinpath(f"imu_static_moment_features_{subject_id}_{night_id}.csv")
    endpoint_file = export_path.joinpath(f"sleep_endpoints_{subject_id}_{night_id}.csv")

    data = None
    if any([compute_endpoints, compute_features, not feature_file.exists(), not endpoint_file.exists()]):
        try:
            data = subset.imu
        except (
            ImuDataNotFoundException,
            ImuInvalidDateException,
            NoSuitableImuDataFoundException,
            DateNotAvailableException,
            ValueError,
        ) as e:
            date = subset.date.loc[(subject_id, night_id), "date"]
            _handle_processing_log(subset.subject_folder_path, subject_id, date, e)
            return None, None

    if compute_endpoints or not endpoint_file.exists():
        # sleep endpoints are not available or should be overwritten
        endpoints_imu = _compute_endpoints(data, subset.sampling_rate)
        if endpoints_imu is not None:
            endpoints_imu.to_csv(endpoint_file)
    else:
        endpoints_imu = pd.read_csv(endpoint_file)

    if compute_features or not feature_file.exists():
        # features are not available or should be overwritten
        if kwargs.get("compare_endpoints", False):
            # compare imu-based endpoints with self-report endpoints
            endpoint_list = [endpoints_imu, endpoints_selfreport]
            endpoint_types = ["imu", "selfreport"]
        else:
            if endpoints_selfreport is not None:
                endpoint_list = [endpoints_selfreport]
                endpoint_types = ["selfreport"]
            else:
                endpoint_list = [endpoints_imu]
                endpoint_types = ["imu"]

        dict_static_moments = {}
        for endpoints, kind in zip(endpoint_list, endpoint_types):
            if endpoints is None:
                continue
            sleep_onset, wake_onset = _get_endpoints(endpoints, kind)
            df_features = _compute_features(
                data,
                sleep_onset=sleep_onset,
                wake_onset=wake_onset,
                sampling_rate=subset.sampling_rate,
                **kwargs,
            )
            if df_features is None:
                continue
            dict_static_moments[kind] = df_features

        sm_features = pd.concat(dict_static_moments, names=["wakeup_type"])
        sm_features.to_csv(feature_file)
    else:
        sm_features = load_long_format_csv(feature_file)

    return endpoints_imu, sm_features


def _compute_endpoints(data: pd.DataFrame, sampling_rate: float):
    sleep_results = predict_pipeline_acceleration(data, sampling_rate=sampling_rate, sleep_wake_scale_factor=0.1)
    if sleep_results is None:
        return None
    return endpoints_as_df(sleep_results["sleep_endpoints"])


def _compute_features(
    data: pd.DataFrame,
    sleep_onset: Union[str, pd.Timestamp],
    wake_onset: Union[str, pd.Timestamp],
    **kwargs,
) -> Optional[pd.DataFrame]:
    if sleep_onset is np.nan or wake_onset is np.nan:
        return None

    if all([isinstance(s, str) for s in [sleep_onset, wake_onset]]):
        data_sleep = data.between_time(sleep_onset, wake_onset)
    else:
        data_sleep = data.loc[sleep_onset:wake_onset]

    # sm_total = _compute_static_moment_features(data_sleep, last=None, **kwargs)
    sm_last_hour = _compute_static_moment_features(data_sleep, last="60min", **kwargs)
    sm_last_30min = _compute_static_moment_features(data_sleep, last="30min", **kwargs)

    sm_dict = {}
    # if sm_total is not None:
    #    sm_dict["total"] = sm_total
    if sm_last_hour is not None:
        sm_dict["last_hour"] = sm_last_hour
    if sm_last_30min is not None:
        sm_dict["last_30min"] = sm_last_30min

    if len(sm_dict) == 0:
        return None
    return pd.concat(sm_dict, names=["time_span", "imu_feature"])


def _compute_static_moment_features(data: pd.DataFrame, last: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]:
    if last:
        data = data.last(last)
    data_gyr = data.filter(like="gyr")
    data_acc = data.filter(like="acc")
    sm = find_static_moments(
        data_gyr,
        threshold=kwargs.get("thres"),
        window_sec=kwargs.get("window_sec"),
        overlap_percent=kwargs.get("overlap"),
        sampling_rate=kwargs.get("sampling_rate"),
    )
    features = compute_features(data_acc, sm)
    if features is None:
        return None
    features = features.T
    features.columns = ["data"]
    return features


def _get_endpoints(
    endpoints: pd.DataFrame, endpoint_type: str
) -> Union[Tuple[pd.Timestamp, pd.Timestamp], Tuple[str, str]]:
    if endpoint_type == "imu":
        sleep_onset = pd.Timestamp(endpoints.iloc[0]["sleep_onset"])
        wake_onset = pd.Timestamp(endpoints.iloc[0]["wake_onset"])
    elif endpoint_type == "selfreport":
        sleep_onset = endpoints.iloc[0]["sleep_onset_selfreport"]
        wake_onset = endpoints.iloc[0]["wake_onset_selfreport"]
    else:
        sleep_onset = None
        wake_onset = None
    return sleep_onset, wake_onset


def _handle_processing_log(folder_path: path_t, subject_id: str, date: pd.Timestamp, error_type: Exception) -> None:
    if isinstance(error_type, ImuDataNotFoundException):
        return
    file_path = folder_path.joinpath(f"processing_log_{subject_id}.log")
    log_dict = {}
    if file_path.exists():
        log_dict = json.loads(file_path.read_text())
    log_dict[str(date.date())] = str(error_type)
    file_path.write_text(json.dumps(log_dict, indent=4))
