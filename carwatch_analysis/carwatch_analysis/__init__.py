# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

from carwatch_analysis import imu_helper, saliva_helper, general_helper

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "carwatch_analysis"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
