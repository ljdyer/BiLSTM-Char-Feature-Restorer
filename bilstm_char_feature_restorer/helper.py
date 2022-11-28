import os
import pickle
from typing import Any, Union

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from uniseg.graphemecluster import grapheme_clusters

try:
    from IPython.display import clear_output    # type: ignore
except:  # noqa: E722
    pass

MESSAGE_RAM_IN_USE = "RAM currently in use: {ram_in_use}%"

STR_OR_LIST_OR_SERIES_TYPE_ERROR = \
    "Must have type list, str, or pandas.Series."
STR_OR_LIST_TYPE_ERROR = "Must have type list, str, or pandas.Series."


# ====================
def try_clear_output():

    try:
        clear_output(wait=True)
    except:  # noqa: E722
        pass


# ====================
def show_ram_used():

    print(MESSAGE_RAM_IN_USE.format(
        ram_in_use=psutil.virtual_memory().percent))


# ====================
def len_gclust(str_: str) -> int:
    """Return a the number of grapheme clusters in the string."""

    return len(list_gclust(str_))


# ====================
def list_gclust(str_: str) -> list:
    """Return a list of grapheme clusters in the string."""

    return list(grapheme_clusters(str_))


# ====================
def only_or_all(input_: Union[list, tuple]) -> Any:
    """If the list or tuple contains only a single element, return that
    element.

    Otherwise return the original list or tuple."""

    if len(input_) == 1:
        return input_[0]
    else:
        return input_


# ====================
def str_or_list_or_series_to_list(input_: Union[str, list, pd.Series]) -> list:

    if isinstance(input_, str):
        return [input_]
    elif isinstance(input_, pd.Series):
        return input_.to_list()
    elif isinstance(input_, list):
        return input_
    else:
        raise TypeError(STR_OR_LIST_OR_SERIES_TYPE_ERROR)


# ====================
def str_or_list_to_list(input_: Union[str, list]) -> list:

    if isinstance(input_, str):
        return [input_]
    elif isinstance(input_, list):
        return input_
    else:
        raise TypeError(STR_OR_LIST_TYPE_ERROR)


# ====================
def object_or_list_to_list(input_: Any) -> list:

    if isinstance(input_, list):
        return input_
    else:
        return [input_]


# ====================
def mk_dir_if_does_not_exist(path):

    if not os.path.exists(path):
        os.makedirs(path)


# ====================
def get_tqdm() -> type:
    """Return tqdm.notebook.tqdm if code is being run from a notebook,
    or tqdm.tqdm otherwise"""

    if is_running_from_ipython():
        tqdm_ = notebook_tqdm
    else:
        tqdm_ = non_notebook_tqdm
    return tqdm_


# ====================
def is_running_from_ipython():
    """Determine whether or not the current script is being run from
    a notebook"""

    try:
        # Notebooks have IPython module installed
        from IPython import get_ipython  # type: ignore   # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


# ====================
def display_or_print(obj: Any):

    if is_running_from_ipython():
        display(obj)    # type: ignore   # noqa: F821
    else:
        print(obj)


# ====================
def load_file(fp: str, mmap: bool = False):
    """Load a .pickle file, or .npy file with mmap_mode either True or False"""

    _, fext = os.path.splitext(fp)
    if fext == '.pickle':
        return load_pickle(fp)
    elif fext == '.npy' and mmap is False:
        return load_npy(fp)
    elif fext == '.npy' and mmap is True:
        return np.load(fp, mmap_mode='r')
    else:
        raise RuntimeError('Invalid file ext!')


# ====================
def load_pickle(fp: str) -> Any:
    """Load a .pickle file and return the data"""

    with open(fp, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


# ====================
def load_npy(fp: str) -> Any:
    """Load a .npy file and return the data"""

    with open(fp, 'rb') as f:
        opened = np.load(f)
    return opened


# ====================
def save_file(data: Any, fp: str):
    """Save data to a .pickle or .npy file"""

    _, fext = os.path.splitext(fp)
    if fext == '.pickle':
        save_pickle(data, fp)
    elif fext == '.npy':
        save_npy(data, fp)
    else:
        raise RuntimeError('Invalid file ext!')


# ====================
def save_pickle(data: Any, fp: str):
    """Save data to a .pickle file"""

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


# ====================
def save_npy(data: Any, fp: str):
    """Save data to a .npy file"""

    with open(fp, 'wb') as f:
        np.save(f, data)


# ====================
def display_dict(dict_: dict):

    df = pd.DataFrame.from_dict(dict_, orient='index')
    display_or_print(df)
