import pandas as pd
import os
import sys
import importlib
from f.load_mx.matrix import BaseMatrix
from pathlib import Path
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

# print(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class FinlabDataMatrix(BaseMatrix):
    def __init__(self, universe_num, start_date, end_date, time_granularity, matrix_cache_folder, data_folder):
        self.matrix_type = "FinlabMatrix"

        from f.load_mx.universe import get_universe
        from f.load_mx.time_list import get_time_list
        universe = get_universe()  
        self.time_list = get_time_list()
        self.universe = universe
        super().__init__(universe, start_date, end_date, time_granularity, matrix_cache_folder, self.time_list)
        

        self.dl = None
        from dataloader.finlab_loader import DataSelector, API

        # finlab_path = Path(data_folder)/'finlab_loader'

        # # Instantiate the data loader with the path
        self.dl = DataSelector()

    '''
    Input args
        target_enum
        start_time 預設是全拿 
        end_time 預設是全拿
    '''        
    def get_finlab_type_data(self, target_enum, start_time=None, end_time=None, how=None, on=None):
        
        df = self.dl.select(what=target_enum, 
                            start_date=start_time, 
                            end_date=end_time,
                            universe=self.universe,
                            on=on,
                            how=how)
        df = pd.DataFrame(df)

        # Check if any column contains string data
        has_string_data = any(df[col].dtype in ['object', 'string'] for col in df.columns)
        
        # Only convert to float if there's no string data
        if not has_string_data:
            df = df.astype(float)
        
        # Update underlying_df while maintaining its structure
        self.underlying_df.update(df, overwrite=True)
        
        return self
    
    def dump_underlying_matrix(
        self, dump_path=Path(PROJECT_ROOT)/'data'/"finlab_underlying_matrix"/"underlying_matrix.csv"
    ):
        self.underlying_df.to_csv(dump_path)


if __name__ == "__main__":
    import f
    f.set_runtime(
        g_start_time='20200101',
        g_end_time='20241231',
        g_time_granularity='4h',
        g_use_cache=False,
        g_matrix_cache_folder='test-cache',
        g_data_folder = 'data',
        g_blow_cache = False,
        g_debug_mode = True,
        g_universe_num = 0
    )

    print('path', Path('../data/').resolve())
    flmatrix = FinlabDataMatrix(1, "20170817", "20240630", "1d",'test-cache', "data" )
    print('finish')
    
    # cctxMatrix = FinlabDataMatrix(
    #     "20170817", "20240630", "4h", "4h-data/", "../data"
    # )

    # cctxMatrix.get_futures_binance_data("o")