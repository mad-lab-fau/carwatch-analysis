from pathlib import Path
from typing import Sequence, Tuple, Union, Any

import pandas as pd

import biopsykit as bp
from biopsykit.utils import path_t

import matplotlib.pyplot as plt


def subject_id_from_path(path: path_t) -> str:
    return str(path.name).split("_")[0]


def process_dataset(data: pd.DataFrame, fs: int, subject_id: str, night_id: int, export_figures: bool,
                    plot_export_path: path_t, **kwargs) -> Union[Tuple[None, None], Tuple[pd.DataFrame, Any]]:
    from biopsykit.signals.imu import get_static_sequences
    from biopsykit.sleep.sleep_endpoints import endpoints_as_df
    from biopsykit.signals.imu.feature_extraction.static_sequences import static_sequence_features

    night_id += 1
    sleep_results = bp.sleep.sleep_endpoints.predict_pipeline(data, sampling_rate=fs, sleep_wake_scale_factor=0.1)
    if not sleep_results:
        print("Subject {} - Night {} does not contain any sleep data!".format(subject_id, night_id))
        return None, None

    sleep_endpoints = sleep_results['sleep_endpoints']

    data = bp.sleep.sleep_endpoints.cut_to_wear_block(data, sleep_results['major_wear_block'])

    if export_figures:
        # turn off interactive mode for saving figures
        plt.ioff()
        fig, ax = bp.sleep.plotting.sleep_imu_plot(data, datastreams=['acc', 'gyr'], sleep_endpoints=sleep_endpoints,
                                                   downsample_factor=10 * fs, figsize=(10, 8))
        fig.tight_layout()
        bp.utils.export_figure(fig=fig, filename="sleep_{}_{}".format(subject_id, night_id), base_dir=plot_export_path,
                               formats=['pdf'], use_subfolder=False)
        plt.close(fig)
        # turn interactive mode on again
        plt.ion()

    data_sleep = data.loc[sleep_endpoints['sleep_onset']:sleep_endpoints['wake_onset']]
    data_gyr = data_sleep.filter(like="gyr")
    data_acc = data_sleep.filter(like="acc")
    ss = get_static_sequences(data_gyr, threshold=kwargs.get('thres'),
                              window_samples=kwargs.get('window_size'),
                              overlap_samples=kwargs.get('overlap'))
    ss_features = static_sequence_features(data_acc, ss, start=sleep_endpoints['sleep_onset'],
                                           end=sleep_endpoints['wake_onset'])

    data_last_hour = data_sleep.last("1h")
    data_gyr = data_last_hour.filter(like="gyr")
    data_acc = data_last_hour.filter(like="acc")

    ss_last = get_static_sequences(data_gyr, threshold=kwargs.get('thres'),
                                   window_samples=kwargs.get('window_size'),
                                   overlap_samples=kwargs.get('overlap'))
    ss_features_last = static_sequence_features(data_acc, ss_last, start=data_last_hour.index[0],
                                                end=data_last_hour.index[-1])
    ss_features_last.columns = [c + '_last_hour' for c in ss_features_last.columns]
    ss_features = ss_features.join(ss_features_last)

    df_endpoints = endpoints_as_df(sleep_endpoints, subject_id)

    return df_endpoints, ss_features


def process_subject(subject_dir: path_t, load_raw: bool, export_figures: bool, feature_export_path: path_t,
                    sleep_endpoints_export_path: path_t, plot_export_path: path_t, **kwargs):
    from tqdm.notebook import tqdm
    subject_dir = Path(subject_dir)

    subject_id = subject_id_from_path(subject_dir)
    nilspod_files = sorted(subject_dir.glob("*.bin"))

    list_features_subject = []
    list_sleep_endpoints = []

    # check whether old processing results already exist
    feature_files = sorted(feature_export_path.glob("chronotype_features_{}.csv".format(subject_id)))
    endpoint_files = sorted(sleep_endpoints_export_path.glob("sleep_endpoints_{}.csv".format(subject_id)))

    # load raw data if no old processing results exist or if data should be overwritten
    if load_raw or len(feature_files) == 0 or len(endpoint_files) == 0:

        if len(nilspod_files) == 0:
            return None, None

        night_id = 0
        for file in tqdm(nilspod_files, desc=subject_id, leave=False):
            try:
                data, fs = bp.io.nilspod.load_dataset_nilspod(file)
            except ValueError:
                continue

            df_endpoints, ss_features = process_dataset(data, fs, subject_id, night_id,
                                                        export_figures=export_figures,
                                                        plot_export_path=plot_export_path, **kwargs)

            if df_endpoints is not None:
                list_sleep_endpoints.append(df_endpoints)
                list_features_subject.append(ss_features)

        if len(list_features_subject) == 0:
            return None, None

        df_features_subject = concat_nights(list_features_subject)
        df_features_subject.to_csv(feature_export_path.joinpath("chronotype_features_{}.csv".format(subject_id)))

        df_endpoints_subject = concat_nights(list_sleep_endpoints)
        df_endpoints_subject.to_csv(sleep_endpoints_export_path.joinpath("sleep_endpoints_{}.csv".format(subject_id)))

    else:
        df_features_subject = pd.read_csv(feature_files[0], index_col=['night'])
        df_endpoints_subject = pd.read_csv(endpoint_files[0], index_col=['night'])

    return df_endpoints_subject, df_features_subject


def concat_nights(list_df: Sequence[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(list_df, ignore_index=True)
    df.index.name = "night"
    return df
