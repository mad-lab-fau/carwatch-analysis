import numpy as np
import pandas as pd
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.utils.time import extract_time_from_filename

from carwatch_analysis._types import path_t
from carwatch_analysis.exceptions import NoSuitableImuDataFoundException


def _load_closest_nilspod_recording_for_date(folder_path: path_t, date: pd.Timestamp) -> pd.DataFrame:
    """
    Find the closest NilsPod recording for a given date.
    """

    imu_files = sorted(folder_path.glob("*.bin"))
    recording_date = date + pd.Timedelta("1d")

    # extract dates from imu files
    nilspod_dates = [
        extract_time_from_filename(file, r"NilsPodX-\w{4}_(\w+).bin", "%Y%m%d_%H%M%S") for file in imu_files
    ]
    # compute difference between actual recording date and dates of NilsPod recordings
    date_diffs = [np.abs(recording_date - date) for date in nilspod_dates]
    # the closest date needs to be within 10 hours of the recording date, if not, throw an error
    if min(date_diffs) > pd.Timedelta("10h"):
        raise NoSuitableImuDataFoundException(f"No suitable IMU recording found for date '{date.date()}'.")

    imu_file = imu_files[np.argmin(date_diffs)]
    return load_dataset_nilspod(imu_file)[0]
