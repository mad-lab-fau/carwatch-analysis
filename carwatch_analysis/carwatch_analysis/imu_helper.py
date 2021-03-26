from pathlib import Path
from typing import Union, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import biopsykit as bp
from _types import path_t
from biopsykit.stats import StatsPipeline
from carwatch_analysis.general_helper import subject_id_from_path, concat_nights


def analysis_imu_features(data: pd.DataFrame, variable: str,
                          test_type: Optional[str] = 'welch_anova') -> StatsPipeline:
    if test_type == 'welch_anova':
        posthoc = 'pairwise_gameshowell'
    else:
        posthoc = 'pairwise_ttests'

    steps = [
        ('prep', 'normality'),
        ('prep', 'equal_var'),
        ('test', test_type),
        ('posthoc', posthoc)
    ]
    params = {
        'groupby': 'feature',
        'group': variable,
        'dv': 'data',
        'between': variable,
        'padjust': 'bonf'
    }
    pipeline = StatsPipeline(
        steps=steps,
        params=params
    )

    pipeline.apply(data)
    return pipeline


def process_subject(subject_dir: path_t, compute_endpoints: bool, compute_features: bool, export_figures: bool,
                    feature_export_path: path_t, sleep_endpoints_export_path: path_t, plot_export_path: path_t,
                    **kwargs):
    from tqdm.notebook import tqdm
    subject_dir = Path(subject_dir)

    subject_id = subject_id_from_path(subject_dir)
    nilspod_files = sorted(subject_dir.glob("*.bin"))

    list_features = []
    list_sleep_endpoints = []

    df_selfreport_endpoints = kwargs.get('selfreport_endpoints', None)

    # check whether old processing results already exist
    feature_files = sorted(feature_export_path.glob("imu_features_{}.csv".format(subject_id)))
    endpoint_files = sorted(sleep_endpoints_export_path.glob("sleep_endpoints_{}.csv".format(subject_id)))

    night_id = 0
    for file in tqdm(nilspod_files, desc=subject_id, leave=False):
        night_id += 1

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
            df_endpoints = pd.read_csv(endpoint_files[0], index_col=['night'])
            if night_id - 1 in df_endpoints.index:
                sleep_endpoints = df_endpoints.loc[[night_id - 1]]
            else:
                continue

        selfreport_endpoints = None
        if df_selfreport_endpoints is not None and night_id - 1 in df_selfreport_endpoints.index:
            selfreport_endpoints = df_selfreport_endpoints.loc[[night_id - 1]]

        if compute_features or len(feature_files) == 0:
            # features are not available or should be overwritten
            if kwargs.get('compare_endpoints', False):
                ep_list = [sleep_endpoints, selfreport_endpoints]
                ep_types = ['imu', 'selfreport']
            else:
                if selfreport_endpoints is not None:
                    ep_list = [selfreport_endpoints]
                    ep_types = ['selfreport']
                else:
                    ep_list = [sleep_endpoints]
                    ep_types = ['imu']

            dict_ss = {}
            for endpoints, kind in zip(ep_list, ep_types):
                if endpoints is None:
                    continue
                sleep_onset, wake_onset = _get_endpoints(endpoints, kind)
                ss_features = _compute_features(data, sleep_onset, wake_onset, **kwargs)
                if ss_features is None:
                    continue
                dict_ss[kind] = ss_features

            ss_features = pd.concat(dict_ss, axis=1, names=['type'])
        else:
            df_features = pd.read_csv(feature_files[0], index_col=0, header=[0, 1, 2])
            print(df_features)
            if night_id - 1 in df_features.index:
                ss_features = df_features.loc[[night_id - 1]]
            else:
                continue

        if sleep_endpoints is not None:
            list_sleep_endpoints.append(sleep_endpoints)
        if ss_features is not None:
            list_features.append(ss_features)

    df_endpoints_subject = concat_nights(list_sleep_endpoints)
    df_features_subject = concat_nights(list_features)

    if df_endpoints_subject is not None:
        df_endpoints_subject.to_csv(
            sleep_endpoints_export_path.joinpath("sleep_endpoints_{}.csv".format(subject_id)))
    if df_features_subject is not None:
        df_features_subject.to_csv(feature_export_path.joinpath("imu_features_{}.csv".format(subject_id)))

    return df_endpoints_subject, df_features_subject

    # load raw data if no old processing results exist or if data should be overwritten
    # if load_raw or len(feature_files) == 0 or len(endpoint_files) == 0:
    #
    #     if len(nilspod_files) == 0:
    #         return None, None
    #
    #     night_id = 0
    #     for file in tqdm(nilspod_files, desc=subject_id, leave=False):
    #         night_id += 1
    #         try:
    #             data, fs = bp.io.nilspod.load_dataset_nilspod(file)
    #         except ValueError:
    #             continue
    #         df_endpoints, df_features = process_dataset(data, fs, subject_id, night_id,
    #                                                     export_figures=export_figures,
    #                                                     plot_export_path=plot_export_path, **kwargs)
    #
    #         if df_endpoints is not None:
    #             list_sleep_endpoints.append(df_endpoints)
    #         if df_features is not None:
    #             list_features_subject.append(df_features)
    #
    #     df_endpoints_subject = concat_nights(list_sleep_endpoints)
    #     df_features_subject = concat_nights(list_features_subject)
    #     if df_endpoints_subject is not None:
    #         df_endpoints_subject.to_csv(
    #             sleep_endpoints_export_path.joinpath("sleep_endpoints_{}.csv".format(subject_id)))
    #     if df_features_subject is not None:
    #         df_features_subject.to_csv(feature_export_path.joinpath("imu_features_{}.csv".format(subject_id)))
    #
    # else:
    #     df_features_subject = pd.read_csv(feature_files[0], index_col=['night'])
    #     df_endpoints_subject = pd.read_csv(endpoint_files[0], index_col=['night'])

    # return df_endpoints_subject, df_features_subject


def _compute_endpoints(data: pd.DataFrame, fs: int, subject_id: str, night_id: int, export_figures: bool,
                       plot_export_path: path_t):
    from biopsykit.sleep.sleep_endpoints import endpoints_as_df

    sleep_results = bp.sleep.sleep_endpoints.predict_pipeline(data, sampling_rate=fs, sleep_wake_scale_factor=0.1)
    if not sleep_results:
        print("Subject {} - Night {} does not contain any sleep data!".format(subject_id, night_id))
        return None, None

    sleep_endpoints = sleep_results['sleep_endpoints']

    data = bp.sleep.sleep_endpoints.cut_to_wear_block(data, sleep_results['major_wear_block'])

    if export_figures:
        # turn off interactive mode for saving figures
        plt.ioff()
        fig, ax = bp.sleep.plotting.sleep_imu_plot(data, datastreams=['acc', 'gyr'],
                                                   sleep_endpoints=sleep_endpoints,
                                                   downsample_factor=10 * fs, figsize=(10, 8))
        fig.tight_layout()
        bp.utils.export_figure(fig=fig, filename="sleep_{}_{}".format(subject_id, night_id),
                               base_dir=plot_export_path,
                               formats=['pdf'], use_subfolder=False)
        plt.close(fig)
        # turn interactive mode on again
        plt.ion()

    df_endpoints = endpoints_as_df(sleep_endpoints, subject_id)
    return df_endpoints


def _compute_features(data: pd.DataFrame, sleep_onset: Union[str, pd.Timestamp], wake_onset: Union[str, pd.Timestamp],
                      **kwargs):
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
        ss_dict['total'] = ss_total
    if ss_last_hour is not None:
        ss_dict['last_hour'] = ss_last_hour
    if ss_last_30min is not None:
        ss_dict['last_30min'] = ss_last_30min

    if len(ss_dict) == 0:
        return None
    ss_features = pd.concat(ss_dict, axis=1, names=['time_span', 'feature'])
    return ss_features


def _static_features(data: pd.DataFrame, last: Optional[str] = None, **kwargs):
    from biopsykit.signals.imu.feature_extraction.static_sequences import get_static_sequences, static_sequence_features

    if last:
        data = data.last(last)
    data_gyr = data.filter(like="gyr")
    data_acc = data.filter(like="acc")
    ss = get_static_sequences(data_gyr, threshold=kwargs.get('thres'),
                              window_samples=kwargs.get('window_size'),
                              overlap_samples=kwargs.get('overlap'))
    ss_features = static_sequence_features(data_acc, ss)
    return ss_features


def _get_endpoints(endpoints: pd.DataFrame, endpoint_type: str) -> Union[
    Tuple[pd.Timestamp, pd.Timestamp], Tuple[str, str]]:
    if endpoint_type == 'imu':
        sleep_onset = pd.Timestamp(endpoints.iloc[0]['sleep_onset'])
        wake_onset = pd.Timestamp(endpoints.iloc[0]['wake_onset'])
    elif endpoint_type == 'selfreport':
        sleep_onset = endpoints.iloc[0]['sleep_onset_selfreport']
        wake_onset = endpoints.iloc[0]['wake_onset_selfreport']
    else:
        sleep_onset = None
        wake_onset = None
    return sleep_onset, wake_onset
