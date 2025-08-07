from ..load_mx.matrix_polygon import PolygonDataMatrix
from ..set_runtime import load_store_from_cache
from tools.load_mx.matrix import BaseMatrix
from tools.time_list import get_time_list
import tools
import warnings
import pandas as pd
import logging
logger = logging.getLogger(__name__) 

@load_store_from_cache
def Prices(column = 'c', runtime_vars=None):
    """
    Arguments:
        column (str): The OHLCV column to specify. Options are:
            - 'o' for open
            - 'h' for high
            - 'l' for low
            - 'c' for close
            - 'v' for volume
        runtime_vars (RuntimeVars, optional): Runtime variables containing configuration for fetching data.

    """
    polygon_matrix = PolygonDataMatrix(
            runtime_vars.g_start_time,
            runtime_vars.g_end_time,
            runtime_vars.g_time_granularity,
            runtime_vars.g_matrix_cache_folder,
            runtime_vars.g_data_folder
    )
    prices = polygon_matrix.get_ohlcv(
        column=column
    )
    return prices

@load_store_from_cache
def IntradayReturn(high_threshold=0.09, low_threshold=-0.09, runtime_vars=None):
    """
    Gets the intraday returns using open prices.
    """
    ret = f.Prices('c')/f.Prices('o')-1
    
    # TODO：需要有其他方式來評估權重不得調整
    # last_day_ret = f.Lag(1, ret)
    # full_ret = f.Prices('o')/(f.Lag(1, f.Prices('o'))-1)
    # full_ret = full_ret.underlying_df
    # ret = ret.underlying_df
    # last_day_ret = last_day_ret.underlying_df
    # ret[ret>high_threshold] = full_ret[ret>high_threshold]
    # ret[ret<low_threshold] = full_ret[ret<low_threshold]

    universe = get_universe()
    time_list = get_time_list()
    bm = BaseMatrix(universe, runtime_vars.g_start_time, runtime_vars.g_end_time, runtime_vars.g_time_granularity, runtime_vars.g_matrix_cache_folder, time_list)
    bm.underlying_df = ret
    return bm


@load_store_from_cache
def Returns(time_delta = -1, runtime_vars=None):
    """
    Gets the returns using closed prices.
    # 日期代表該日賣出的return

    Arguments:
        time_delta (int): -1 by default, which returns the return of a time tick.

    Returns:
        BaseMatrix: The return calculated using the closed price and time_delta.

    Warning:
        A positive time_delta will have look-ahead-bias. Do not use this in your signal in production!
    """
    if time_delta > 0:
        logger.warning("A positive time_delta will have look-ahead bias. Do not use this in your signal in production!")
    
    '''
    Returns不需extended，因爲訊號生成當下最快可以賣出就是明天，因爲無法往前
    '''
    curr_prices = f.Prices('c')
    curr_prices = curr_prices.ffill()

    # TODO: Have delsited values filled properly(非自願下市)

    curr_prices = f.ListingNanAlign(curr_prices) # filter out the over filling entries
    delta_prices = curr_prices.shift(-time_delta) # days(postive value) before

    return curr_prices / delta_prices - 1


## 開一個 finlab load matrix 之類的把customable的全部API寫一個界面
@load_store_from_cache
def FinlabType(target_type, on=None, how=None, runtime_vars=None):
    if runtime_vars.g_universe_num == 1:
        finlab_matrix = FinlabDataMatrix(
                runtime_vars.g_universe_num,
                runtime_vars.g_start_time,
                runtime_vars.g_end_time,
                runtime_vars.g_time_granularity,
                runtime_vars.g_matrix_cache_folder,
                runtime_vars.g_data_folder
            )
        twstock = finlab_matrix.get_finlab_type_data(
                    target_type,
                    start_time=runtime_vars.g_start_time,
                    end_time=runtime_vars.g_end_time,
                    on=on,
                    how=how
        )
        return twstock

@load_store_from_cache
def Empty(runtime_vars=None):
    return BaseMatrix(get_universe(), runtime_vars.g_start_time, runtime_vars.g_end_time, runtime_vars.g_time_granularity, runtime_vars.g_matrix_cache_folder, get_time_list())
