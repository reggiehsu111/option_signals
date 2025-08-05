import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hashlib import sha256
import inspect
import operator

# import traceback
# import logging
# logger = logging.getLogger(__name__)

def hash_df(df):
    data_bytes = df.to_numpy().tobytes()
    hash_obj = sha256(data_bytes)
    return hash_obj


class BaseMatrix:
    """
    The Base class for matrix.
    For each matrix, we have a unique signature that is computed with the arguments and the runtime variables feed into the matrix.
    
    Attributes:
        matrix_type (str): The type of the matrix.
        universe (list): A list of tickers. For Binance matrix, the universe is stored in a json file that is parsed using the CoinMarketCap API.
        start_time (str): Start time of the matrix.
        end_time (str): End time of the matrix.
        time_granularity (str): Time granularity of the time axis.
        matrix_cache_folder (str): Path to store the matrix. It will create a new folder if the folder does not exist.

    """
    def __init__(self, universe, start_time, end_time, time_granularity, matrix_cache_folder, time_list):
        self.matrix_type = 'BaseMatrix'
        self.universe = pd.Index(universe) if not isinstance(universe, pd.Index) else universe
        self.start_time = start_time
        self.end_time = end_time
        self.time_granularity = time_granularity
        # to make the direct modification of underlying_df go pass getattr to update the signature
        self._underlying_df = pd.DataFrame(np.nan, index=time_list, columns=universe)
        self.matrix_cache_folder = matrix_cache_folder
        self.time_list = pd.Index(time_list) if not isinstance(time_list, pd.Index) else time_list
        self.unique_str = '_'.join([
            self.matrix_type,
            self.start_time,
            self.end_time,
            # WIP: maybe should add the universe and time_list to the signature
        ])

    @property
    def underlying_df(self) -> pd.DataFrame:
        return self._underlying_df
    
    @underlying_df.setter
    def underlying_df(self, value):
        self._underlying_df = value
        # self.update_signature() # to make the direct modification of underlying_df go pass getattr to update the signature

    def __getattr__(self, name):
        if hasattr(self.underlying_df, name):
            attr = getattr(self.underlying_df, name)
            if name in ['loc', 'iloc']:  # Handle loc, iloc directly
                class PropertyProxy:
                    def __getitem__(self_, *args, **kwargs):
                        return attr.__getitem__(*args, **kwargs)
                    def __setitem__(self_, *args, **kwargs):
                        attr.__setitem__(*args, **kwargs)
                return PropertyProxy()
            if isinstance(attr, property):  # Check for properties
                result = attr.fget(self.underlying_df)
                class PropertyProxy:
                    def __getitem__(self_, *args, **kwargs):
                        return result.__getitem__(*args, **kwargs)
                    def __setitem__(self_, *args, **kwargs):
                        result.__setitem__(*args, **kwargs)
                return PropertyProxy()
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    # Check for the 'inplace' parameter in the method signature
                    try:
                        sig = inspect.signature(attr)
                        if 'inplace' in sig.parameters and kwargs.get('inplace', False): #and not self.underlying_df.equals(result): # comment them because inplace always return None which will never be equal 
                            # self.underlying_df = result
                            return self
                        if 'inplace' not in sig.parameters and result is None and name in ['update']: # update has no inplace paramter but is actually doing inplace transformation
                            # self.update_signature()
                            return self
                    except ValueError:
                        # Handle the case where no signature is found
                        pass
                    bm = BaseMatrix(self.universe, self.start_time, self.end_time, self.time_granularity, self.matrix_cache_folder, self.time_list)
                    bm.underlying_df = result
                    # self.update_signature()
                    return bm
                return wrapper
            return attr

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def get_matrix_coverage(self):
        """
        Calculates and stores the actual coverage of the matrix.
        """
        ax = self.underlying_df.count(axis=1).plot()
        fig = ax.get_figure()
        fig.savefig('coverage.pdf')

        fig, ax = plt.subplots(ncols=4, nrows=int(len(self.underlying_df.columns)/4)+1)
        for i, x in enumerate(self.underlying_df.columns):
            self.underlying_df[x].plot(ax=ax[int(i/4), i%4])
        fig.set_size_inches(18, 36)
        fig.savefig('coverage_ind.pdf', dpi=100)

    def update_signature(self):
        df_bytes = self.underlying_df.to_numpy().tobytes()
        self.signature = sha256(self.unique_str.encode('utf-8') + df_bytes).hexdigest()

        return self.signature

    # Access all the methods from Pandas dataframes
    def __getitem__(self, key):
        # Safe access: handle Series vs DataFrame separately
        if isinstance(self.underlying_df, pd.Series):
            if isinstance(key, int):
                return self.underlying_df.iloc[key]  # avoid future warning
            else:
                return self.underlying_df.loc[key]   # safe label-based access
        return self.underlying_df[key]

    def __setitem__(self, key, value):
        self.underlying_df[key] = value
        # self.update_signature()

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.update_signature()


    def __repr__(self):
        return repr(self.underlying_df)


    # Helper method to apply a binary operation and return a Matrix instance
    def _apply_op(self, other, op_func):
        if isinstance(other, BaseMatrix):
            other = other.underlying_df  # Use the underlying DataFrame if `other` is a Matrix
        if isinstance(other, (pd.Series, pd.Index)) and other.shape[0] == self.underlying_df.shape[0]:
            other = pd.DataFrame(np.tile(other.values, (self.underlying_df.shape[1], 1)).T, 
                                 columns=self.underlying_df.columns, 
                                 index=self.underlying_df.index)
        if isinstance(other, (np.ndarray)) and other.shape[0] == self.underlying_df.shape[0]:
            other = pd.DataFrame(np.tile(other, (self.underlying_df.shape[1], 1)).T, 
                                 columns=self.underlying_df.columns, 
                                 index=self.underlying_df.index)
   
        bm = BaseMatrix(self.universe, self.start_time, self.end_time, self.time_granularity, self.matrix_cache_folder, self.time_list)  # Wrap the result in a new Matrix
        bm.underlying_df = op_func(self.underlying_df, other).replace([np.inf, -np.inf], np.nan)
        return bm

    # Arithmetic operators
    def __add__(self, other): return self._apply_op(other, operator.add)
    def __sub__(self, other): return self._apply_op(other, operator.sub)
    def __mul__(self, other): return self._apply_op(other, operator.mul)
    def __floordiv__(self, other): return self._apply_op(other, operator.floordiv)
    def __mod__(self, other): return self._apply_op(other, operator.mod)
    def __pow__(self, other): return self._apply_op(other, operator.pow)

    # Comparison operators
    def __eq__(self, other): return self._apply_op(other, operator.eq)
    def __ne__(self, other): return self._apply_op(other, operator.ne)
    def __lt__(self, other): return self._apply_op(other, operator.lt)
    def __le__(self, other): return self._apply_op(other, operator.le)
    def __gt__(self, other): return self._apply_op(other, operator.gt)
    def __ge__(self, other): return self._apply_op(other, operator.ge)

    # Reverse arithmetic operators for cases where scalar is on the left side
    def __radd__(self, other): return self._apply_op(other, operator.add)
    def __rsub__(self, other): return self._apply_op(other, operator.sub)
    def __rmul__(self, other): return self.__mul__(other)
    def __rtruediv__(self, other): return self._apply_op(other, operator.truediv)
    def __rfloordiv__(self, other): return self._apply_op(other, operator.floordiv)
    def __rmod__(self, other): return self._apply_op(other, operator.mod)
    def __rpow__(self, other): return self._apply_op(other, operator.pow)

    def __neg__(self):
        negated_data = -self.underlying_df
        bm = BaseMatrix(self.universe, self.start_time, self.end_time, self.time_granularity, self.matrix_cache_folder, self.time_list)
        bm.underlying_df = negated_data
        return bm
    def __and__(self, other):
        if isinstance(other, BaseMatrix):
            result_data = self.underlying_df & other.underlying_df
        elif isinstance(other, pd.DataFrame):
            result_data = self.underlying_df & other
        else:
            raise TypeError(f"Unsupported operand type(s) for &: 'BaseMatrix' and '{type(other)}'")
        bm = BaseMatrix(self.universe, self.start_time, self.end_time, self.time_granularity, self.matrix_cache_folder, self.time_list)
        bm.underlying_df = result_data 
        return bm

    def __truediv__(self, other):
        bm = BaseMatrix(self.universe, self.start_time, self.end_time, self.time_granularity, self.matrix_cache_folder, self.time_list)
        if isinstance(other, BaseMatrix):
            # Divide by another BaseMatrix instance, ignoring NaNs
            if self.underlying_df.shape != other.underlying_df.shape:
                raise ValueError("DataFrames must have the same shape for element-wise division.")
            divisor_df = other.underlying_df.fillna(0)
            result_df = self.underlying_df.div(divisor_df)
            result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            bm.underlying_df = result_df
            return bm
        elif isinstance(other, pd.DataFrame):
            # Divide by a pandas DataFrame, ignoring NaNs
            if self.underlying_df.shape != other.shape:
                raise ValueError("DataFrames must have the same shape for element-wise division.")
            divisor_df = other.fillna(0)
            result_df = self.underlying_df.div(divisor_df)
            result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            bm.underlying_df = result_df
            return bm
        elif isinstance(other, (int, float)):
            # Divide by a scalar
            result_df = self.underlying_df / other
            bm.underlying_df = result_df
            return bm
        else:
            return self._apply_op(other, operator.truediv)

    def __len__(self):
        return len(self.underlying_df)
