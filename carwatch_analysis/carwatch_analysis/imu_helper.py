from pathlib import Path
from typing import Union, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import biopsykit as bp
from biopsykit.signals.imu.static_moment_detection import find_static_moments
from biopsykit.signals.imu.wear_detection import WearDetection
import biopsykit.signals.imu.feature_extraction.static_moments as sm
from biopsykit.stats import StatsPipeline

from carwatch_analysis._types import path_t
from carwatch_analysis.general_helper import subject_id_from_path, concat_nights


def analysis_imu_features(data: pd.DataFrame, variable: str, test_type: Optional[str] = "welch_anova") -> StatsPipeline:
    if test_type == "welch_anova":
        posthoc = "pairwise_gameshowell"
    else:
        posthoc = "pairwise_ttests"

    steps = [
        ("prep", "normality"),
        ("prep", "equal_var"),
        ("test", test_type),
        ("posthoc", posthoc),
    ]
    params = {
        "groupby": "imu_feature",
        "dv": "data",
        "between": variable,
        "padjust": "bonf",
    }
    pipeline = StatsPipeline(steps=steps, params=params)

    pipeline.apply(data)
    return pipeline


def process_subject(
    subject_dir: path_t,
    compute_endpoints: bool,
    compute_features: bool,
    export_figures: bool,
    feature_export_path: path_t,
    sleep_endpoints_export_path: path_t,
    plot_export_path: path_t,
    **kwargs,
):
    from tqdm.notebook import tqdm

    subject_dir = Path(subject_dir)

    subject_id = subject_id_from_path(subject_dir)
    nilspod_files = list(sorted(subject_dir.glob("*.bin")))

    dict_features = {}
    dict_sleep_endpoints = {}

    df_selfreport_endpoints = kwargs.get("selfreport_endpoints", None)

    # check whether old processing results already exist
    feature_files = sorted(feature_export_path.glob("imu_features_{}.csv".format(subject_id)))
    endpoint_files = sorted(sleep_endpoints_export_path.glob("sleep_endpoints_{}.csv".format(subject_id)))

    for night_id, file in enumerate(tqdm(nilspod_files, desc=subject_id, leave=False)):

        data = None
        fs = None

        if any([compute_endpoints, compute_features, len(feature_files) == 0, len(endpoint_files) == 0]):
            try:
                data, fs = bp.io.nilspod.load_dataset_nilspod(file)
            except ValueError:
                continue

        if compute_endpoints or len(endpoint_files) == 0:
            # sleep endpoints are not available or should be overwritten
            sleep_endpoints = _compute_endpoints(data, fs, subject_id, night_id, export_figures, plot_export_path)
        else:
            df_endpoints = pd.read_csv(endpoint_files[0], index_col=["night"])
            if night_id in df_endpoints.index:
                sleep_endpoints = df_endpoints.loc[[night_id]]
            else:
                continue

        selfreport_endpoints = None
        if df_selfreport_endpoints is not None and night_id in df_selfreport_endpoints.index:
            selfreport_endpoints = df_selfreport_endpoints.loc[[night_id]]

        if compute_features or len(feature_files) == 0:
            # features are not available or should be overwritten
            if kwargs.get("compare_endpoints", False):
                ep_list = [sleep_endpoints, selfreport_endpoints]
                ep_types = ["imu", "selfreport"]
            else:
                if selfreport_endpoints is not None:
                    ep_list = [selfreport_endpoints]
                    ep_types = ["selfreport"]
                else:
                    ep_list = [sleep_endpoints]
                    ep_types = ["imu"]

            dict_ss = {}
            for endpoints, kind in zip(ep_list, ep_types):
                if endpoints is None:
                    continue
                sleep_onset, wake_onset = _get_endpoints(endpoints, kind)
                ss_features = _compute_features(data, sleep_onset, wake_onset, **kwargs)
                if ss_features is None:
                    continue
                dict_ss[kind] = ss_features

            ss_features = pd.concat(dict_ss, axis=1, names=["wakeup_type"])
        else:
            df_features = pd.read_csv(feature_files[0])
            df_features = df_features.set_index(list(df_features.columns[:-1]))

            if night_id in df_features.index:
                ss_features = df_features.xs(night_id, level="night")
            else:
                continue

        if sleep_endpoints is not None:
            dict_sleep_endpoints[night_id] = sleep_endpoints
        if ss_features is not None:
            dict_features[night_id] = ss_features

    df_endpoints_subject = None
    df_features_subject = None
    if len(dict_sleep_endpoints) > 0:
        df_endpoints_subject = pd.concat(dict_sleep_endpoints.values())
    if len(dict_features) > 0:
        df_features_subject = pd.concat(dict_features, names=["night"])

    if df_endpoints_subject is not None:
        df_endpoints_subject.to_csv(sleep_endpoints_export_path.joinpath("sleep_endpoints_{}.csv".format(subject_id)))
    if df_features_subject is not None:
        df_features_subject.to_csv(feature_export_path.joinpath("imu_features_{}.csv".format(subject_id)))

    return df_endpoints_subject, df_features_subject


def _compute_endpoints(
    data: pd.DataFrame,
    sampling_rate: float,
    subject_id: str,
    night_id: int,
    export_figures: bool,
    plot_export_path: path_t,
):
    from biopsykit.sleep.sleep_endpoints import endpoints_as_df

    sleep_results = bp.sleep.sleep_processing_pipeline.predict_pipeline_acceleration(
        data, sampling_rate=sampling_rate, sleep_wake_scale_factor=0.1
    )
    if not sleep_results:
        print("Subject {} - Night {} does not contain any sleep data!".format(subject_id, night_id))
        return None, None

    sleep_endpoints = sleep_results["sleep_endpoints"]

    data = WearDetection.cut_to_wear_block(data, sleep_results["major_wear_block"])

    if export_figures:
        # turn off interactive mode for saving figures
        plt.ioff()
        fig, ax = bp.sleep.plotting.sleep_imu_plot(
            data,
            datastreams=["acc", "gyr"],
            sleep_endpoints=sleep_endpoints,
            downsample_factor=int(10 * sampling_rate),
            figsize=(10, 8),
        )
        fig.tight_layout()
        bp.utils.file_handling.export_figure(
            fig=fig,
            filename="sleep_{}_{}".format(subject_id, night_id),
            base_dir=plot_export_path,
            formats=["pdf"],
            use_subfolder=False,
        )
        plt.close(fig)
        # turn interactive mode on again
        plt.ion()

    df_endpoints = endpoints_as_df(sleep_endpoints, subject_id)
    return df_endpoints


def _compute_features(
    data: pd.DataFrame, sleep_onset: Union[str, pd.Timestamp], wake_onset: Union[str, pd.Timestamp], **kwargs
):
    if sleep_onset is np.nan or wake_onset is np.nan:
        return None

    if all([isinstance(s, str) for s in [sleep_onset, wake_onset]]):
        data_sleep = data.between_time(sleep_onset, wake_onset)
    else:
        data_sleep = data.loc[sleep_onset:wake_onset]

    ss_total = _static_features(data_sleep, last=None, **kwargs)
    ss_last_hour = _static_features(data_sleep, last="60min", **kwargs)
    ss_last_30min = _static_features(data_sleep, last="30min", **kwargs)

    ss_dict = {}
    if ss_total is not None:
        ss_dict["total"] = ss_total
    if ss_last_hour is not None:
        ss_dict["last_hour"] = ss_last_hour
    if ss_last_30min is not None:
        ss_dict["last_30min"] = ss_last_30min

    if len(ss_dict) == 0:
        return None
    ss_features = pd.concat(ss_dict, axis=1, names=["time_span", "imu_feature"])
    return ss_features


def _static_features(data: pd.DataFrame, last: Optional[str] = None, **kwargs):
    if last:
        data = data.last(last)
    data_gyr = data.filter(like="gyr")
    data_acc = data.filter(like="acc")
    ss = find_static_moments(
        data_gyr,
        threshold=kwargs.get("thres"),
        window_samples=kwargs.get("window_size"),
        overlap_samples=kwargs.get("overlap"),
    )
    ss_features = sm.compute_features(data_acc, ss)
    return ss_features


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
