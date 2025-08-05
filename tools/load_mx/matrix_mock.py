import pandas as pd
import os
import sys
import importlib
from tools.load_mx.matrix import BaseMatrix
from pathlib import Path
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# print(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class MockMatrix(BaseMatrix):
    def __init__(
        self, start_date, end_date, time_granularity, matrix_cache_folder, data_folder
    ):
        self.matrix_type = "MockMatrix"

        from tools.load_mx.universe import get_universe
        from tools.load_mx.time_list import get_time_list

        universe = get_universe()
        self.time_list = get_time_list()
        self.universe = universe
        super().__init__(
            universe,
            start_date,
            end_date,
            time_granularity,
            matrix_cache_folder,
            self.time_list,
        )

        self.dl = None
        from data.mock_dataloader import MockDataloader

        self.dl = MockDataloader(start_date, end_date, time_granularity)

        return self.dl.generate()


if __name__ == "__main__":
    import tools
