import os
import time

import pandas as pd
from scikit-learn.model_selection import ParameterGrid, train_test_split

from bilstm_char_feature_restorer.helper import (display_dict,
                                                 display_or_print, load_file,
                                                 mk_dir_if_does_not_exist,
                                                 save_file, try_clear_output)

GRID_SEARCH_ATTRS_FNAME = 'GRID_SEARCH_ATTRS.pickle'

GRID_SEARCH_LOG_DF_COLS = [
    'model_name', 'units', 'batch_size', 'dropout', 'recur_dropout',
    'keep_size', 'val_size', 'epoch', 'loss', 'accuracy', 'val_loss',
    'val_accuracy', 'Time', 'Exception'
]

MESSAGE_SAVED_GRID_SEARCH = """\
Saved BiLSTMCharFeatureRestorerGridSearch with the below attributes to \
{root_folder}"""
MESSAGE_LOADED_GRID_SEARCH = """\
Loaded BiLSTMCharFeatureRestorerGridSearch with the below attributes from \
{root_folder}"""
MESSAGE_SKIPPING_PARAMS = """\
Skipping parameter combination at index {i} because results \
are already in the grid search log.
"""

ERROR_GRID_SEARCH_EXISTS = """\
This BiLSTMCharFeatureRestorer already has a grid search with the name \
{grid_search_name}. Either continue the existing grid search, or choose a \
new grid search name."""


# ====================
class BiLSTMCharFeatureRestorerGridSearch:

    # ====================
    def __init__(self, parent, **kwargs):

        self.parent = parent
        self.__dict__.update(kwargs)
        grid_search_path = self.root_folder()
        if os.path.exists(grid_search_path):
            raise ValueError(
                ERROR_GRID_SEARCH_EXISTS.format(
                    grid_search_name=self.grid_search_name)
            )
        else:
            mk_dir_if_does_not_exist(grid_search_path)
        self.set_param_combos()
        self.train_val_split()
        self.save()
        self.run_grid_search()

    # ====================
    def train_val_split(self):

        X = self.parent.get_asset('X', mmap=True)
        all_idxs = range(len(X))
        if self.keep_size != 1.0:
            keep_idxs, _ = \
                train_test_split(all_idxs, test_size=(1.0-self.keep_size))
        else:
            keep_idxs = all_idxs
        self.train_idxs, self.val_idxs = \
            train_test_split(keep_idxs, test_size=self.val_size)

    # ====================
    def save(self):

        attrs = self.__dict__.copy()
        attrs_path = self.attrs_path()
        save_file(attrs, attrs_path)
        print(MESSAGE_SAVED_GRID_SEARCH.format(root_folder=self.root_folder()))
        self.show_attrs()

    # ====================
    def attrs_path(self):

        return os.path.join(self.root_folder(), GRID_SEARCH_ATTRS_FNAME)

    # ====================
    def show_attrs(self):

        attrs = self.__dict__.copy()
        display_dict(attrs)

    # ====================
    def root_folder(self):

        return os.path.join(
            self.parent.grid_search_path(), self.grid_search_name
        )

    # ====================
    @classmethod
    def load(cls, parent, grid_search_name: str):

        self = cls.__new__(cls)
        self.parent = parent
        self.grid_search_name = grid_search_name
        attrs = load_file(
            os.path.join(self.root_folder(), GRID_SEARCH_ATTRS_FNAME)
        )
        self.__dict__.update(attrs)
        print(
            MESSAGE_LOADED_GRID_SEARCH.format(root_folder=self.root_folder())
        )
        self.show_attrs()
        self.run_grid_search()
        return self

    # ====================
    def set_param_combos(self):

        self.param_combos = list(ParameterGrid({
            'units': self.units,
            'batch_size': self.batch_size,
            'dropout': self.dropout,
            'recur_dropout': self.recur_dropout
        }))
        self.save()

    # ====================
    def run_grid_search(self):

        log_df = self.get_log_df()
        for i, parameters in enumerate(self.param_combos):
            try_clear_output()
            display_or_print(log_df)
            model_name = self.model_name(i)
            if len(log_df[log_df['model_name'] == model_name]) > 0:
                print(MESSAGE_SKIPPING_PARAMS.format(i=i))
                continue
            model_args = {
                'model_name': self.model_name(i),
                'units': parameters['units'],
                'batch_size': parameters['batch_size'],
                'dropout': parameters['dropout'],
                'recur_dropout': parameters['recur_dropout'],
                'val_size': self.val_size,
                'keep_size': self.keep_size
            }
            print(model_args)
            try:
                self.parent.add_model(
                    **model_args, overwrite=True, supress_save_msg=True
                )
                self.parent.model.train_idxs = self.train_idxs
                self.parent.model.val_idxs = self.val_idxs
                start_time = time.time()
                self.parent.model.train(self.epochs)
                gs_row = {**model_args, 'Time': time.time() - start_time}
                model_log_df = pd.read_csv(self.parent.model.log_path())
                last_epoch_info = model_log_df.iloc[-1].to_dict()
                gs_row = {**gs_row, **last_epoch_info}
            except Exception as e:
                gs_row = {**model_args, 'Exception': e}
            log_df = log_df.append(gs_row, ignore_index=True)
            self.save_log(log_df)
        try_clear_output()
        display_or_print(log_df)

    # ====================
    def grid_search_folder(self):

        gs_folder = os.path.join(
            self.root_folder(),
            'grid_searches',
            self.grid_search_name
        )
        mk_dir_if_does_not_exist(gs_folder)
        return gs_folder

    # ====================
    def log_path(self) -> str:

        return os.path.join(self.root_folder(), 'log.csv')

    # ====================
    def get_log_df(self) -> pd.DataFrame:

        log_path = self.log_path()
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
        else:
            log_df = pd.DataFrame(columns=GRID_SEARCH_LOG_DF_COLS)
            log_df.to_csv(log_path, index=False)
        return log_df

    # ====================
    def save_log(self, log: pd.DataFrame):

        log.to_csv(self.log_path(), index=False)

    # ====================
    def show_log(self):

        display_or_print(self.get_log_df())

    # ====================
    def model_name(self, i: int) -> str:

        return f"{self.grid_search_name}_{i}"

    # ====================
    def show_max(self, col: str = 'val_accuracy'):
        """Display the row from the grid search df for which the value in the
        column specified is maximized.

        Args:
          col (str, optional):
            The column to maximize. Defaults to 'val_accuracy'.
        """

        log_df = self.get_log_df()
        col_vals = log_df[col].to_list()
        max_val = max(col_vals)
        max_row = log_df[log_df[col] == max_val]
        display_or_print(max_row)
