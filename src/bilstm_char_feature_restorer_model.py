import os
from random import shuffle

import keras
import pandas as pd
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from helper.misc import (display_dict, load_file, mk_dir_if_does_not_exist,
                         save_file, display_or_print)

MODEL_ATTRS_FNAME = 'MODEL_ATTRS.pickle'
MODEL_LOG_FNAME = 'log.csv'

MESSAGE_LOADED_MODEL = """\
Loaded BiLSTMCharFeatureRestorerModel with the below attributes from \
{root_folder}"""
MESSAGE_SAVED_MODEL = """\
Saved BiLSTMCharFeatureRestorerModel with the below attributes to \
{root_folder}"""

ERROR_MODEL_EXISTS = """\
This BiLSTMCharFeatureRestorer already has a model with the name \
{model_name}. Either load the existing model, or choose a new model name."""
ERROR_TRAIN_OR_VAL = 'Parameter train_or_val must be "TRAIN" or "VAL".'


# ====================
class BiLSTMCharFeatureRestorerModel:

    # ====================
    def __init__(self, parent, **kwargs):

        self.parent = parent
        self.__dict__.update(kwargs)
        model_path = self.root_folder()
        if kwargs['overwrite'] is False and os.path.exists(model_path):
            raise ValueError(
                ERROR_MODEL_EXISTS.format(model_name=self.model_name)
            )
        else:
            mk_dir_if_does_not_exist(model_path)
        self.train_val_split()
        self.save()
        self.model = self.new_model()
        self.model.save(self.latest_path())

    # ====================
    def save(self):

        attrs = self.__dict__.copy()
        if 'model' in attrs:
            del attrs['model']
        attrs_path = self.attrs_path()
        save_file(attrs, attrs_path)
        if self.supress_save_msg is False:
            print(MESSAGE_SAVED_MODEL.format(root_folder=self.root_folder()))
            self.show_attrs()

    # ====================
    @classmethod
    def load(cls, parent, model_name: str):

        self = cls.__new__(cls)
        self.parent = parent
        self.model_name = model_name
        attrs = load_file(os.path.join(self.root_folder(), MODEL_ATTRS_FNAME))
        self.__dict__.update(attrs)
        print(MESSAGE_LOADED_MODEL.format(root_folder=self.root_folder()))
        self.show_attrs()
        self.model = keras.models.load_model(self.latest_path())
        return self

    # ====================
    def load_checkpoint(self, checkpoint_number: int):

        self.model = keras.models.load_model(
            self.checkpoint_path(checkpoint_number)
        )

    # ====================
    def last_epoch(self):

        try:
            log_df = pd.read_csv(self.log_path())
            last_epoch_ = max([int(e) for e in log_df['epoch'].to_list()]) + 1
        except FileNotFoundError:
            last_epoch_ = 0
        return last_epoch_

    # ====================
    def latest_path(self):

        return os.path.join(self.root_folder(), 'latest')

    # ====================
    def checkpoint_path(self, checkpoint_number: int):

        return os.path.join(self.root_folder(), f'cp-{checkpoint_number:02}')

    # ====================
    def log_path(self):

        return os.path.join(self.root_folder(), MODEL_LOG_FNAME)

    # ====================
    def attrs_path(self):

        return os.path.join(self.root_folder(), MODEL_ATTRS_FNAME)

    # ====================
    def root_folder(self):

        return os.path.join(self.parent.models_path(), self.model_name)

    # ====================
    def show_attrs(self):

        attrs = self.__dict__.copy()
        display_dict(attrs)

    # ====================
    def new_model(self):

        num_X_categories, num_y_categories = \
            self.parent.get_num_categories(['X_TOKENIZER', 'Y_TOKENIZER'])
        model = Sequential()
        model.add(Bidirectional(
                    LSTM(
                        self.units,
                        return_sequences=True,
                        dropout=self.dropout,
                        recurrent_dropout=self.recur_dropout
                    ),
                    input_shape=(
                        self.parent.seq_length,
                        num_X_categories + 1
                    )
                ))
        if num_y_categories == 2:
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation = 'softmax'
            loss = 'categorical_crossentropy'
        print(f'activation: {activation}; loss: {loss};')
        model.add(TimeDistributed(Dense(
            num_y_categories + 1, activation=activation
        )))
        model.compile(
            loss=loss,
            optimizer='adam',
            metrics=['accuracy']
        )
        return model

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
    def train(self, epochs: int):
        """Train the model.

        Required arguments:
        -------------------
        epochs: int             The number of epochs to train for.
        """

        last_epoch = self.last_epoch()
        num_train = len(self.train_or_val_idxs('TRAIN'))
        num_val = len(self.train_or_val_idxs('VAL'))
        batch_size = self.batch_size
        save_each_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.root_folder(), 'cp-{epoch:02d}'),
            save_freq='epoch'
        )
        save_latest_checkpoint = ModelCheckpoint(
            filepath=self.latest_path(),
            save_freq='epoch'
        )
        csv_logger = CSVLogger(self.log_path(), append=True)
        self.model.fit(
            self.data_loader('TRAIN'),
            steps_per_epoch=(num_train // batch_size),
            validation_data=self.data_loader('VAL'),
            validation_steps=(num_val // batch_size),
            callbacks=[
                save_each_checkpoint, save_latest_checkpoint, csv_logger
            ],
            initial_epoch=last_epoch,
            epochs=(last_epoch + epochs),
        )

    # ====================
    def train_or_val_idxs(self, train_or_val: str):

        if train_or_val == 'TRAIN':
            return self.train_idxs
        elif train_or_val == 'VAL':
            return self.val_idxs
        else:
            raise RuntimeError(ERROR_TRAIN_OR_VAL)

    # ====================
    def data_loader(self, train_or_val: str):
        """Iterator function to create batches"""

        batch_size = self.batch_size
        while True:
            X = self.parent.get_asset('X', mmap=True)
            y = self.parent.get_asset('Y', mmap=True)
            idxs = self.train_or_val_idxs(train_or_val)
            shuffle(idxs)
            num_iters = len(idxs) // batch_size
            num_X_categories, num_y_categories = \
                self.parent.get_num_categories(['X_TOKENIZER', 'Y_TOKENIZER'])
            for i in range(num_iters):
                X_encoded = to_categorical(
                    X[idxs[(i*batch_size):((i+1)*batch_size)]],
                    num_X_categories + 1
                )
                y_encoded = to_categorical(
                    y[idxs[(i*batch_size):((i+1)*batch_size)]],
                    num_y_categories + 1
                )
                yield (X_encoded, y_encoded)

    # ====================
    def show_log(self):

        log_file_df = pd.read_csv(self.log_path())
        pd.set_option('display.max_rows', 100)
        display_or_print(log_file_df)
        pd.reset_option('display.max_rows')

    # ====================
    def get_log_df(self):

        return pd.read_csv(self.log_path())

    # ====================
    def show_max(self, col: str = 'val_accuracy'):

        log_df = self.get_log_df()
        col_vals = log_df[col].to_list()
        max_val = max(col_vals)
        max_row = log_df[log_df[col] == max_val]
        display_or_print(max_row)
