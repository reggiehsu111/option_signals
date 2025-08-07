from tools.set_runtime import load_store_from_cache

import pandas as pd
import logging
from typing import Union

# Get the logger using the module name
logger = logging.getLogger(__name__)

@load_store_from_cache
def get_time_list(runtime_vars=None):
    start_time = None if pd.isna(runtime_vars.g_start_time) else pd.Timestamp(runtime_vars.g_start_time)
    end_time = pd.Timestamp.today().normalize() if pd.isna(runtime_vars.g_end_time) else pd.Timestamp(runtime_vars.g_end_time)
    return pd.date_range(start=start_time, end=end_time, freq=runtime_vars.g_time_granularity)