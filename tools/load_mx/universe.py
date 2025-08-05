from dataloader.finlab_loader.utils.get_universe import get_universe as finlab_get_universe
from dataloader.market_type_generator import get_market_type_df as generator_market_type
from f.set_runtime import load_store_from_cache
from typing import List, Optional
import pandas as pd

def get_universe_from_df(
    universe_type: List[str] = ['sii', 'otc', 'tib'],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.Index:
    """
    Select stocks from the pivoted market_type_df that match universe_type
    in the specified date range.
    """
    start_date = pd.Timestamp('2007-01-01') if start_date is None else pd.Timestamp(start_date)
    end_date = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
    df:pd.DataFrame = generator_market_type(end_date=end_date, tib=True)

    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered = df.loc[mask]

    # Find stocks whose values ever appear in the universe_type during the range
    matched = filtered.isin(universe_type)
    selected = matched.any(axis=0)

    return selected[selected].index.sort_values()

@load_store_from_cache  
def get_universe(runtime_vars=None):
    try: 
        universe = get_universe_from_df(start_date=runtime_vars.g_start_time, end_date=runtime_vars.g_end_time)
        return universe
        
    except Exception as e:
        print(f"[Fallback] Failed to use new market type generator: {e}")
    
    # Fallback to Finlab
    universe = finlab_get_universe(['sii', 'otc'], runtime_vars.g_start_time, runtime_vars.g_end_time)
    return universe


