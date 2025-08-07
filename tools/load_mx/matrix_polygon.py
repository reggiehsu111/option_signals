import pandas as pd
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
        self.underlying_df = self.dl.generate(
            self.start_time, self.end_time, src=column
        )
        return self.underlying_df
