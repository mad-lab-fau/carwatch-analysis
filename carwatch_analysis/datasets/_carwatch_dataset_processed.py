"""Dataset representing raw data of the CARWatch dataset."""
import re
from functools import lru_cache
from typing import Optional, Sequence, Union, Dict

import biopsykit as bp
import pandas as pd
from biopsykit.io import load_long_format_csv, load_questionnaire_data
from tpcp import Dataset

from carwatch_analysis._types import path_t
from carwatch_analysis.datasets._utils import _load_app_logs
from carwatch_analysis.exceptions import ImuDataNotFoundException, AppLogDataNotFoundException

_cached_load_app_logs = lru_cache(maxsize=5)(_load_app_logs)


class CarWatchDatasetProcessed(Dataset):
    """Representation of processed data collected during the CARWatch study.

    Processed data contain sleep endpoints, IMU static period features, saliva, self-reports, as well as metadata.

    Data are only loaded once the respective attributes are accessed.

    Parameters
    ----------
    base_path
        The base folder where the dataset can be found.

    """

    base_path: path_t
    use_cache: bool

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: Optional[bool] = True,
    ):
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)
        self.base_path = base_path
        self.use_cache = use_cache

    def create_index(self) -> pd.DataFrame:
        return self._load_condition_map().reset_index()[["subject", "night"]]

    def _load_condition_map(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("questionnaire/processed/condition_map.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                f"File {data_path.name} not found! Please run the notebook 'Questionnaire_Processing.ipynb' first!"
            )
        return pd.read_csv(data_path, index_col=["subject", "night"])

    @property
    def date(self) -> pd.DataFrame:
        """Return the recording date.

        Returns
        -------
        :class:`~pandas.Timestamp`
            date of recording

        """
        cort_data = self.cortisol_samples
        date_date = cort_data[["date"]].droplevel("sample").reset_index().drop_duplicates()
        date_date = date_date.set_index(["subject", "night", "condition"])
        return date_date.apply(pd.to_datetime)

    @property
    def condition_map(self) -> pd.DataFrame:
        """Load and return mapping of conditions to subjects and nights."""
        data = self._load_condition_map()
        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return data.loc[(subject_ids, nights), :]

    @property
    def codebook(self):
        """Return the codebook of the questionnaire data."""
        data_path = self.base_path.joinpath("questionnaire/raw")
        return bp.io.load_codebook(data_path.joinpath("Codebook_CARWatch.xlsx"))

    @property
    def questionnaire(self) -> pd.DataFrame:
        """Load and return questionnaire data."""
        if self.is_single("night"):
            raise ValueError("Questionnaire data can not be accessed for a single night!")
        data_path = self.base_path.joinpath("questionnaire/processed").joinpath("questionnaire_data.csv")
        data = pd.read_csv(data_path)
        data = data.set_index(["subject", "night", "condition"])

        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    @property
    def chronotype_bedtime(self) -> pd.DataFrame:
        """Load and return chronotype and bedtime information."""
        data_path = self.base_path.joinpath("questionnaire/processed/chronotype_bedtimes.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                f"File {data_path.name} not found! Please run the notebook 'Questionnaire_Processing.ipynb' first!"
            )
        data = load_questionnaire_data(data_path, subject_col="subject", additional_index_cols=["night"])
        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return data.loc[(subject_ids, nights), :]

    @property
    def endpoints_selfreport(self) -> pd.DataFrame:
        """Return the self-reported sleep endpoints."""
        return self.chronotype_bedtime[["sleep_onset_selfreport", "bed_selfreport", "wake_onset_selfreport"]]

    @property
    def sleep_information_merged(self) -> pd.DataFrame:
        """Return sleep information, merged from self-reports and IMU data."""
        data_path = self.base_path.joinpath("questionnaire/processed/sleep_information_merged.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                f"File {data_path.name} not found! Please run the notebook 'Sleep_Information_Merge.ipynb' first!"
            )
        data = pd.read_csv(data_path)
        data = data.set_index(["subject", "night", "condition"])
        for time_col in ["sleep_onset_time", "bed_time", "wake_onset_time"]:
            data[time_col] = pd.to_timedelta(data[time_col])

        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return data.loc[(subject_ids, nights), :]

    @property
    def subject_folder_path(self) -> path_t:
        """Return path to folder of one participant.

        Returns
        -------
        :class:`~pathlib.Path`
            path to folder containing IMU data

        """
        if not self.is_single("subject"):
            raise ValueError("subject folder path can only be accessed for a single participant!")
        subject_id = self.index["subject"][0]

        return self.base_path.joinpath(f"sleep/{subject_id}")

    @property
    def imu_static_moment_features(self) -> pd.DataFrame:
        """Load and return sleep endpoints computed from IMU data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Dataframe with IMU-based sleep endpoints of one night

        """
        if not self.is_single(None):
            raise ValueError("IMU data can only be accessed for a single participant and a single night!")

        subject_id = self.index["subject"][0]
        night = self.index["night"][0]
        data_path = self.subject_folder_path.joinpath("processed")
        file_name = data_path.joinpath(f"imu_static_moment_features_{subject_id}_{night}.csv")
        if not file_name.exists():
            raise ImuDataNotFoundException(
                f"File {file_name.name} does not exist – either IMU data is unavailable "
                "or IMU features were not computed yet!"
            )
        return load_long_format_csv(file_name)

    @property
    def imu_sleep_endpoints(self) -> pd.DataFrame:
        """Load and return sleep endpoints computed from IMU data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Dataframe with IMU-based sleep endpoints of one night

        """
        if not self.is_single(None):
            raise ValueError("IMU data can only be accessed for a single participant and a single night!")

        subject_id = self.index["subject"][0]
        night = self.index["night"][0]
        data_path = self.subject_folder_path.joinpath("processed")
        file_name = data_path.joinpath(f"sleep_endpoints_{subject_id}_{night}.csv")
        if not file_name.exists():
            raise ImuDataNotFoundException(
                f"File {file_name.name} does not exist – either IMU data is unavailable "
                "or sleep endpoints were not computed yet!"
            )
        return pd.read_csv(file_name, index_col=["date"])

    @property
    def cortisol_samples(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("saliva/processed")
        data = pd.read_csv(data_path.joinpath("cortisol_samples.csv"))
        data = data.set_index(["subject", "night", "condition", "sample"])

        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        data["time_abs"] = pd.to_timedelta(data["time_abs"])
        data["wake_onset_time"] = pd.to_timedelta(data["wake_onset_time"])
        data["date"] = pd.to_datetime(data["date"])

        return data.loc[(subject_ids, nights), :]

    @property
    def cortisol_features(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("saliva/processed")
        data = load_long_format_csv(data_path.joinpath("cortisol_features.csv"))

        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return data.loc[(subject_ids, nights), :]

    @property
    def subjects_with_app(self) -> pd.Index:
        """Return a series with the subjects that have an app."""
        folder_path = self.base_path.joinpath("app_logs/cleaned_manual")
        subject_ids = sorted(folder_path.glob("*.csv"))
        subject_ids = [re.findall(r"logs_(\w+).csv", path.name)[0] for path in subject_ids]
        return pd.Index(subject_ids, name="subject")

    @property
    def app_logs(self) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Return the app logs of a participant.

        Returns
        -------

        """
        if self.is_single("night"):
            raise ValueError("App logs can not be accessed for single nights!")
        folder_path = self.base_path.joinpath("app_logs/cleaned_manual")
        subject_ids = self.index["subject"].unique()
        if self.use_cache:
            log_dict = _cached_load_app_logs(folder_path)
        else:
            log_dict = _load_app_logs(folder_path)

        log_dict_out = {}
        for subject_id in subject_ids:
            if subject_id in log_dict:
                log_dict_out[subject_id] = log_dict[subject_id]

        if self.is_single("subject"):
            subject_id = self.index["subject"][0]
            if subject_id not in log_dict_out:
                raise AppLogDataNotFoundException(f"No app logs found for subject {subject_id}!")
            return log_dict_out[subject_id]
        return log_dict_out
