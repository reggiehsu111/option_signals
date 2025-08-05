from dataclasses import dataclass
from hashlib import sha256
import os
import pandas as pd
import pickle
from tools.load_mx.matrix import BaseMatrix
from functools import wraps
from pathlib import Path


def get_project_root():
    """Get the absolute path to the project root directory."""
    # Get the directory containing this file (f/set_runtime.py)
    current_file = Path(__file__)
    # Go up two levels to reach project root (f/set_runtime.py -> f -> project_root)
    return str(current_file.parent.parent)


def resolve_path(path):
    """Convert relative path to absolute path based on project root."""
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return str(Path(get_project_root()) / path)


@dataclass
class RuntimeEnv:
    g_start_time: str
    g_end_time: str
    g_time_granularity: str
    g_use_cache: True
    g_matrix_cache_folder: str
    g_data_folder: str
    g_blow_cache: bool = False
    g_debug_mode: bool = True
    g_universe_num: int = 0


def hash_df(df):
    data_bytes = df.to_numpy().tobytes()
    hash_obj = sha256(data_bytes)
    return hash_obj


def set_runtime(**kwargs):
    # Convert relative paths to absolute paths
    path_keys = ["g_matrix_cache_folder", "g_data_folder"]
    for key in path_keys:
        if key in kwargs:
            kwargs[key] = resolve_path(kwargs[key])

    for key, value in kwargs.items():
        globals()[key] = value


def load_store_from_cache(func):
    # This wrapper function only works when the function returns a single matrix
    @wraps(func)
    def wrapper(*args, **kwargs):
        global_var = globals()
        runtime_vars = RuntimeEnv(
            g_start_time=global_var.get("g_start_time"),
            g_end_time=global_var.get("g_end_time"),
            g_time_granularity=global_var.get("g_time_granularity"),
            g_use_cache=global_var.get("g_use_cache"),
            g_matrix_cache_folder=global_var.get("g_matrix_cache_folder"),
            g_data_folder=global_var.get("g_data_folder"),
            g_blow_cache=global_var.get("g_blow_cache"),
            g_debug_mode=global_var.get("g_debug_mode"),
            g_universe_num=global_var.get("g_universe_num"),
        )
        kwargs["runtime_vars"] = runtime_vars

        # Get the signatures of matrix arguemnts, this is used to load and store the matrices in cache
        matrix_arg_signatures = []
        for _, arg in enumerate(args):
            if isinstance(arg, BaseMatrix):
                matrix_arg_signatures.append(arg.update_signature())
        for _, v in kwargs.items():
            if isinstance(v, BaseMatrix):
                matrix_arg_signatures.append(v.update_signature())

        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        unique_str = "_".join(
            [
                func.__name__,
                runtime_vars.g_start_time,
                runtime_vars.g_end_time,
                runtime_vars.g_time_granularity,
                args_str,
                kwargs_str,
            ]
        )
        matrix_arg_signature_str = "_".join(matrix_arg_signatures)
        signature = sha256(
            unique_str.encode("utf-8") + matrix_arg_signature_str.encode("utf-8")
        ).hexdigest()
        if runtime_vars.g_use_cache:
            pkl_file = (
                runtime_vars.g_matrix_cache_folder
                + "/"
                + func.__name__
                + "_"
                + signature
                + ".pkl"
            )

            if not os.path.isdir(runtime_vars.g_matrix_cache_folder):
                print(
                    "Cache folder does not exist! Creating cache folder: "
                    + runtime_vars.g_matrix_cache_folder
                )
                os.makedirs(runtime_vars.g_matrix_cache_folder)

            if not runtime_vars.g_blow_cache and os.path.exists(pkl_file):
                if runtime_vars.g_debug_mode:
                    print("Read from cache file: " + pkl_file + "...")
                with open(pkl_file, "rb") as file:
                    loaded_mx = pickle.load(file)
                return loaded_mx

        ret_mx = func(*args, **kwargs)
        if runtime_vars.g_use_cache:
            if runtime_vars.g_debug_mode:
                print("Store to cache file: " + pkl_file + "...")
            with open(pkl_file, "wb") as file:
                pickle.dump(ret_mx, file, protocol=4)
        return ret_mx

    return wrapper
