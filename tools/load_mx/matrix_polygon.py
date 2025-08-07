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
        from data.polygon.data_loader import PolygonOHLCDataloader

        self.matrix_type = "Polygon"
        universe = ["I:SPX"]
        self.time_list = self._generate_time_list(start_time, end_time, time_granularity)
        
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

    def _generate_time_list(self, start_time, end_time, time_granularity):
        """
        Generate a time_list with the specified granularity between start_time and end_time.

        Args:
            start_time (str): Start date in 'YYYY-MM-DD'.
            end_time (str): End date in 'YYYY-MM-DD'.
            time_granularity (str): Granularity ('1h', '5m', etc.).

        Returns:
            pd.Index: Time index with appropriate frequency.
        """
        granularity_map = {
            '1h': 'H',
            '5m': '5min',
            '1m': 'min',
            '4h': '4H',
            '1d': 'D'
        }
        if time_granularity not in granularity_map:
            raise ValueError(f"Unsupported time_granularity: {time_granularity}")

        freq = granularity_map[time_granularity]
        time_list = pd.date_range(
            start=start_time,
            end=end_time,
            freq=freq,
            inclusive='both'
        )
        return pd.Index(time_list.tz_localize(None))

    def get_ohlcv(self, column):
        """
        Fetches OHLCV data for the specified column, sets underlying_df with non-NaN values,
        and updates time_list to match the non-NaN timestamps.

        Args:
            column (str): The OHLCV column to fetch (e.g., 'o', 'h', 'l', 'c').

        Returns:
            PolygonDataMatrix: The updated matrix object.
        """
        # Fetch OHLCV data
        ohlcv_df = self.dl.generate(
            self.start_time, self.end_time, src=column
        )

        # Initialize aligned_df with non-NaN data
        if not ohlcv_df.empty:
            # Convert to DataFrame, remove NaN values, and make timestamps timezone-naive
            aligned_df = ohlcv_df.to_frame(name='I:SPX').dropna()
            aligned_df.index = aligned_df.index.tz_localize(None)
            
            # Update time_list to match non-NaN timestamps
            self.time_list = pd.Index(aligned_df.index)
            
            # Set underlying_df
            self.underlying_df = aligned_df
        else:
            # If ohlcv_df is empty, keep empty DataFrame with original time_list
            self.underlying_df = pd.DataFrame(
                np.nan,
                index=self.time_list,
                columns=self.universe
            )


        return self