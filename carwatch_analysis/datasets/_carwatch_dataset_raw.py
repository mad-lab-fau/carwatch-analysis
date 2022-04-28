"""Dataset representing raw data of the CARWatch dataset."""
import itertools
from functools import cached_property, lru_cache
from typing import Optional, Sequence, Tuple

import biopsykit as bp
import pandas as pd
from biopsykit.io import load_questionnaire_data, load_long_format_csv
from biopsykit.utils.dataframe_handling import int_from_str_idx, multi_xs
from tpcp import Dataset

from carwatch_analysis._types import path_t
from carwatch_analysis.datasets._utils import _load_closest_nilspod_recording_for_date
from carwatch_analysis.exceptions import ImuDataNotFoundException, DateNotAvailableException

_cached_load_closest_nilspod_recording_for_date = lru_cache(maxsize=5)(_load_closest_nilspod_recording_for_date)


class CarWatchDatasetRaw(Dataset):
    """Representation of raw data (IMU during sleep, saliva, self-reports) collected during the CARWatch study.

    Data are only loaded once the respective attributes are accessed. By default, the last five calls of the
    ``imu`` attribute are cached. Caching can be disabled by setting the ``use_cache`` argument to ``False`` during
    initialization.

    Parameters
    ----------
    base_path
        The base folder where the dataset can be found.
    use_cache
        ``True`` to cache the last five calls to loading IMU data, ``False`` to disable caching.

    """

    CORRUPTED_SENSORS = ("E901", "1BD7", "C6AA", "2CA4", "5D47")

    base_path: path_t
    use_cache: bool
    _sampling_rate: float = 102.4

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
        quest_path = self.base_path.joinpath("questionnaire/raw")
        quest_data = pd.read_csv(quest_path.joinpath("Questionnaire_Data_CARWatch.csv"))
        index = quest_data["subject"].to_list()
        index = list(itertools.product(index, [0, 1]))
        return pd.DataFrame(index, columns=["subject", "night"])

    @property
    def sampling_rate(self) -> float:
        """Return sampling rate of IMU data in Hz.

        Returns
        -------
        float
            sampling rate in Hz

        """
        return self._sampling_rate

    @property
    def date(self) -> pd.DataFrame:
        """Return the recording date.

        Returns
        -------
        :class:`~pandas.Timestamp`
            date of recording

        """
        subject_ids = self.index["subject"].unique()
        night_ids = self.index["night"].unique()
        data_path = self.base_path.joinpath("questionnaire/raw")
        data = load_questionnaire_data(data_path.joinpath("Questionnaire_Data_CARWatch.csv"))
        data = data.loc[subject_ids]
        data = data[["dateStart"]].join(self.index.set_index(["subject"]))
        data["dateStart"] = pd.to_datetime(data["dateStart"]) + pd.to_timedelta(data["night"], unit="days")
        data = data.set_index(["night"], append=True)
        data.columns = ["date"]
        return data.loc[pd.IndexSlice[:, night_ids], :]

    @cached_property
    def imu(self) -> pd.DataFrame:
        """Load and return IMU data during sleep.

        Returns
        -------
        :class:`~pandas.DataFrame`
            IMU data of one night

        """
        if not self.is_single(None):
            raise ValueError("IMU data can only be accessed for a single participant and night!")

        subject_id = self.index["subject"][0]
        night_id = self.index["night"][0]
        recording_date = self.date.loc[(subject_id, night_id), "date"]

        data_path = self.subject_folder_path.joinpath("raw")
        if not data_path.exists():
            raise ImuDataNotFoundException(f"No IMU recording available for participant {subject_id}!")
        if recording_date is pd.NaT:
            raise DateNotAvailableException(f"No date information available for subject {subject_id}.")

        if self.use_cache:
            return _cached_load_closest_nilspod_recording_for_date(data_path, recording_date)
        return _load_closest_nilspod_recording_for_date(data_path, recording_date)

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
    def codebook(self):
        """Return the codebook of the questionnaire data."""
        data_path = self.base_path.joinpath("questionnaire/raw")
        return bp.io.load_codebook(data_path.joinpath("Codebook_CARWatch.xlsx"))

    @property
    def questionnaire(self):
        """Load and return questionnaire data."""
        if self.is_single("night"):
            raise ValueError("Questionnaire data can not be accessed for a single night!")
        data_path = self.base_path.joinpath("questionnaire/raw")
        data = load_questionnaire_data(data_path.joinpath("Questionnaire_Data_CARWatch.csv"))

        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    @property
    def cortisol(self) -> pd.DataFrame:
        """Load and return cortisol data."""
        data_path = self.base_path.joinpath("questionnaire/raw")
        data = load_questionnaire_data(data_path.joinpath("Questionnaire_Data_CARWatch.csv"))

        cort_long = bp.utils.dataframe_handling.wide_to_long(data, stubname="cort", levels=["night", "sample"])
        cort_long = cort_long.rename(columns={"cort": "cortisol", "cortTime": "time_abs"})
        cort_long = int_from_str_idx(cort_long, "night", r"N(\w)", lambda x: x - 1)

        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return cort_long.loc[(subject_ids, nights), :]

    @property
    def condition_map(self) -> pd.DataFrame:
        """Load and return mapping of conditions to subjects and nights."""
        data_path = self.base_path.joinpath("questionnaire/processed/condition_map.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                f"File {data_path.name} not found! Please run the notebook 'Questionnaire_Processing.ipynb' first!"
            )
        data = pd.read_csv(data_path, index_col=["subject", "night"])
        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return data.loc[(subject_ids, nights), :]

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
        subject_ids = self.index["subject"].unique()
        nights = self.index["night"].unique()

        return data.loc[(subject_ids, nights), :]
