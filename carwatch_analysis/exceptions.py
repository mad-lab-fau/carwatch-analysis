class DateNotAvailableException(Exception):
    """An error indicating that the recording date of the participant is not available."""

    pass


class ImuDataNotFoundException(Exception):
    """An error indicating that no IMU recording is available for the given participant."""

    pass


class NoSuitableImuDataFoundException(Exception):
    """An error indicating that no suitable IMU recording was found for the given date."""

    pass


class ImuInvalidDateException(Exception):
    """An error indicating that the recording date of the IMU data is invalid."""

    pass


class AppLogDataNotFoundException(Exception):
    """An error indicating that no app log data are available for the given participant."""

    pass
