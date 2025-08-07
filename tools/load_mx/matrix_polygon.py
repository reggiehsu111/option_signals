import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import sys
import importlib
from tools.load_mx.matrix import BaseMatrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class PolygonDataMatrix(BaseMatrix):
    def __init__(
        self, start_time, end_time, time_granularity, matrix_cache_folder, data_folder
    ):
        from tools.time_list import get_time_list
        from data.polygon.data_loader import PolygonOHLCDataloader

        self.matrix_type = "Polygon"
        universe = ["I:SPX"]
        self.time_list = get_time_list()
        super().__init__(
            universe,
            start_time,
            end_time,
            time_granularity,
            matrix_cache_folder,
            self.time_list,
        )
        load_dotenv()
        api_key = os.getenv("POLYGON_API_KEY")
        self.dl = PolygonOHLCDataloader(
            api_key=api_key, time_granularity=time_granularity
        )

    def get_ohlcv(self, column):
        """
        Fetches OHLCV data for the specified column and aligns it with the original underlying_df shape,
        mapping values to the nearest prior timestamp in time_list to ensure point-in-time (PIT) data,
        preserving NaN values where no data is available.

        Args:
            column (str): The OHLCV column to fetch (e.g., 'o', 'h', 'l', 'c').

        Returns:
            pd.DataFrame: The aligned underlying_df with fetched data for the specified column.
        """
        # Fetch OHLCV data from PolygonOHLCDataloader
        ohlcv_df = self.dl.generate(
            self.start_time, self.end_time, src=column
        )

        # Create a new DataFrame with the same shape as the original underlying_df
        aligned_df = pd.DataFrame(
            np.nan,
            index=self.time_list,
            columns=self.universe
        )

        # Align fetched data with the time_list index using merge_asof for PIT
        if not ohlcv_df.empty:
            # Convert ohlcv_df to a DataFrame with a column for values and reset index
            ohlcv_df = ohlcv_df.to_frame(name='value').reset_index()
            # Remove timezone from ohlcv_df timestamps to match time_list (assumed naive)
            ohlcv_df['timestamp'] = ohlcv_df['timestamp'].dt.tz_localize(None)

            # Create a DataFrame for time_list
            time_list_df = pd.DataFrame(index=self.time_list).reset_index()
            time_list_df.columns = ['timestamp']

            # Perform merge_asof to map ohlcv_df values to the nearest prior timestamp in time_list
            merged_df = pd.merge_asof(
                time_list_df.sort_values('timestamp'),
                ohlcv_df.sort_values('timestamp'),
                on='timestamp',
                direction='backward',  # Match to the nearest prior timestamp
                allow_exact_matches=True
            )

            # Set the merged values back to aligned_df['I:SPX']
            merged_df = merged_df.set_index('timestamp')
            aligned_df['I:SPX'] = merged_df['value'].reindex(self.time_list, fill_value=np.nan)

        # Set the aligned DataFrame as underlying_df
        self.underlying_df = aligned_df

        return self.underlying_df