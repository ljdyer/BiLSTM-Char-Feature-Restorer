import json
import logging
import os
from random import sample
from typing import Any, List, Union

import numpy as np
import pandas as pd
import psutil
import tensorflow
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from bilstm_char_feature_restorer_grid_search import \
    BiLSTMCharFeatureRestorerGridSearch
from bilstm_char_feature_restorer_model import BiLSTMCharFeatureRestorerModel
from helper.misc import (Int_or_Tuple, Str_or_List, Str_or_List_or_Series,
                         display_dict, display_or_print, get_tqdm, len_gclust,
                         list_gclust, load_file, mk_dir_if_does_not_exist,
                         only_or_all, save_file, show_ram_used,
                         str_or_list_or_series_to_list, str_or_list_to_list)

CLASS_ATTRS_FNAME = 'CLASS_ATTRS.pickle'
MODELS_PATH_NAME = 'models'
GRID_SEARCH_PATH_NAME = 'grid_search'

ASSETS = {
    'CLASS_ATTRS': CLASS_ATTRS_FNAME,
    'X_TOKENIZER': 'X_TOKENIZER.pickle',
    'Y_TOKENIZER': 'Y_TOKENIZER.pickle',
    'X_RAW': 'X_RAW.pickle',
    'Y_RAW': 'Y_RAW.pickle',
    'X_TOKENIZED': 'X_TOKENIZED.pickle',
    'Y_TOKENIZED': 'Y_TOKENIZED.pickle',
    'X': 'X.npy',
    'Y': 'Y.npy',
}

# General messages
SAVED_RAW_SAMPLES = "Saved {num_samples} samples in 'X_RAW' and 'Y_RAW'"
SAVED_TOKENIZER = """Saved tokenizer with {num_categories} \
categories to {tokenizer_name}."""
SAVED_NUMPY_ARRAY = """Saved numpy array with shape {shape} to \
{numpy_asset_name}."""
SAVED_TOKENIZED_SAMPLES = """Saved {num_samples} tokenized samples to \
{tokenized_asset_name}."""

MESSAGE_GRID_SEARCH_EXISTS = """\
There is already a grid search with the name {gs_name}. Attempting to resume \
the existing grid search..."""
MESSAGE_GENERATING_RAW_SAMPLES = "Generating raw samples from data provided..."
MESSAGE_TOKENIZING_INPUTS = "Tokenizing model inputs (X)..."
MESSAGE_TOKENIZING_OUTPUTS = "Tokenizing model outputs (y)..."
MESSAGE_CONVERTING_INPUTS_TO_NUMPY = """Converting model inputs (X) to numpy \
format..."""
MESSAGE_CONVERTING_OUTPUTS_TO_NUMPY = """Converting model outputs (y) to \
numpy format..."""
MESSAGE_LOADED_INSTANCE = """\
Loaded BiLSTMCharFeatureRestorer with the below attributes from \
{root_folder}"""
MESSAGE_SAVED_INSTANCE = """\
Saved BiLSTMCharFeatureRestorer with the below attributes to {root_folder}"""
MESSAGE_SKIPPING_PARAMS = """\
Skipping this parameter combination as it has already been tested in this \
grid search..."""

# Warning messages
WARNING_INPUT_STR_TOO_SHORT = """Warning: length of input string is less \
than the model sequence length."""

# Error messages
ERROR_INPUT_STR_TOO_LONG = """The sequence length for this feature restorer \
is {seq_length} and this input string has {len_input} non-feature \
characters."""
ERROR_ATTRS_AND_LOAD = 'You cannot specify both attrs and load_folder.'
ERROR_MISSING_ATTR = 'attrs must have key {reqd_attr}.'
ERROR_REQD_ATTR_TYPE = "{reqd_attr} should have type {reqd_type}."
ERROR_SPECIFY_ATTRS_OR_LOAD_FOLDER = """You must specify either attrs or \
load_folder."""
ERROR_SPACES_FALSE_NOT_IMPLEMENTED = 'Not implemented yet when spaces=False.'
ERROR_ONE_OF_EACH_FALSE_NOT_IMPLEMENTED = """Not implemented yet when \
one_of_each=False."""
ERROR_CHAR_SHIFT_UNSPECIFIED = """\
char_shift must be specified when spaces=False"""

CHUNKER_NUM_PREFIX_WORDS = 5
CHUNKER_NUM_PREFIX_CHARS = 10

tqdm_ = get_tqdm()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tensorflow.get_logger().setLevel('ERROR')
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)


# ====================
class BiLSTMCharFeatureRestorer:

    # === CLASS INSTANCE ADMIN ===

    # ====================
    def __init__(self,
                 root_folder: str,
                 capitalisation: bool,
                 spaces: bool,
                 other_features: list,
                 seq_length: int,
                 one_of_each: bool = True,
                 char_shift: int = None):
        """Initialize a new instance of the BiLSTMCharFeatureRestorer class

        Required arguments:
        -------------------
        root_folder: str            The folder that will contain information
                                    about the instance to load later, as well
                                    as all assets such as input data and
                                    trained model weights.
                                    This folder will be created if it does not
                                    already exist.

        capitalisation: bool        Whether or not models trained on the
                                    instance will restore capitalisation.

        spaces: bool                Whether or not models trained on the
                                    instance will restore spaces.

        other_features: list        A list of other characters that models
                                    trained on the instance will restore.
                                    E.g. to restore commas and periods, set
                                    other_features=['.', ','].
                                    The order in which the characters appear
                                    in output texts if a single character
                                    possesses more than one feature will be
                                    the same as the order in which they appear
                                    in this list.
                                    Spaces will appear after features in
                                    other_features.

        seq_length: int             The length in characters of model input
                                    sequences used for preprocessing, training,
                                    and prediction.

        one_of_each: bool           If set to True, the model will only
                                    restore a maximum of one of each feature
                                    per character. E.g. '...' will never appear
                                    in output texts.
                                    If set to False, the model will restore an
                                    arbitrary number of each feature (provided
                                    that examples exist in the training data).
                                    (Not implemented yet at the time of
                                    writing.)

        Optional keyword arguments:
        ---------------------------
        char_shift: int = None      Only required if spaces=False.
                                    The step size in characters for the
                                    sliding window when generating input data.
                                    Smaller values of char_shift generate
                                    larger numbers of training examples.


        """

        self.root_folder = root_folder
        mk_dir_if_does_not_exist(self.root_folder)
        self.capitalisation = capitalisation
        self.spaces = spaces
        self.other_features = other_features
        self.seq_length = seq_length
        self.one_of_each = one_of_each
        if self.spaces is False:
            if char_shift is None:
                raise ValueError()
            else:
                self.char_shift = int(char_shift)
        self.set_feature_chars()
        self.save()

    # ====================
    def set_feature_chars(self):
        """Set feature_chars attribute based on other_features and spaces
        attributes"""

        if self.spaces:
            self.feature_chars = self.other_features + [' ']
        else:
            self.feature_chars = self.other_features

    # ====================
    @classmethod
    def load(cls, root_folder: str):
        """Load a saved instance of the BiLSTMCharFeatureRestorer class.

        Required arguments:
        -------------------

        root_folder: str            The root folder that was specified
                                    when the instance was created.
        """

        self = cls.__new__(cls)
        self.root_folder = root_folder
        data = self.get_asset('CLASS_ATTRS')
        self.__dict__.update(data)
        print(MESSAGE_LOADED_INSTANCE.format(root_folder=root_folder))
        self.show_attrs()
        return self

    # ====================
    def save(self):

        attrs = self.__dict__.copy()
        if 'model' in attrs:
            del attrs['model']
        self.save_asset(attrs, 'CLASS_ATTRS')
        print(MESSAGE_SAVED_INSTANCE.format(root_folder=self.root_folder))
        display_dict(attrs)

    # ====================
    def show_attrs(self):
        """Output a table of all the attributes of the class instance"""

        attrs = self.__dict__.copy()
        display_dict(attrs)

    # === MODEL ADMIN ===

    # ====================
    def list_models(self):

        models_path_ = self.models_path()
        for fn in sorted(os.listdir(models_path_)):
            print(fn)

    # ====================
    def models_path(self):

        models_path_ = os.path.join(self.root_folder, MODELS_PATH_NAME)
        mk_dir_if_does_not_exist(models_path_)
        return models_path_

    # ====================
    def add_model(self,
                  model_name: str,
                  units: int,
                  batch_size: int,
                  dropout: float,
                  recur_dropout: float,
                  keep_size: float,
                  val_size: float,
                  overwrite: bool = False,
                  supress_save_msg: bool = False):
        """Create a new BiLSTM model.

        All models assets are saved in the 'models' subfolder of the
        instance root folder, and the 'model' attribute of the current
        instance is set to a BiLSTMCharFeatureRestorerModel object
        representing the currently loaded model.

        Required arguments:
        -------------------
        model_name: str             A name for the new model.
        units: int                  The number of BiLSTM units.
        batch_size: int             The batch size
        dropout: float              The forward dropout rate.
        recur_dropout: float        The recurrent (backward) dropout rate.
        keep_size: float            The proportion of the loaded data to
                                    use in model training.
                                    This will usually be 1.0, but values
                                    such as 0.1 maybe used for grid
                                    searches, etc.
        val_size: float             The proportion of data to use for
                                    validation when training the model.
                                    E.g. set val_size=0.2 for an 80/20
                                    train/val split.
        """

        attrs = locals()
        del attrs['self']
        self.model = BiLSTMCharFeatureRestorerModel(self, **attrs)

    # ====================
    def load_model(self, model_name: str):

        self.model = BiLSTMCharFeatureRestorerModel.load(self, model_name)

    # === ASSET MANAGEMENT ===

    # ====================
    def get_asset(self, asset_name: str, mmap: bool = False) -> Any:
        """Get an asset based on the asset name

        Use numpy memmap if mmap=True"""

        asset_path = self.asset_path(asset_name)
        return load_file(asset_path, mmap=mmap)

    # ====================
    def asset_path(self, asset_name: str) -> str:
        """Get the path to an asset"""

        fname = ASSETS[asset_name]
        return self.get_file_path(fname)

    # ====================
    def get_file_path(self, fname: str) -> str:
        """Get a file path from a file name by appending the root folder of
        the current instance."""

        return os.path.join(self.root_folder, fname)

    # ====================
    def save_asset(self, data: Any, asset_name: str):
        """Save data to the asset file for the named asset"""

        asset_path = self.asset_path(asset_name)
        save_file(data, asset_path)

    # ====================
    def do_assets_exist(self):
        """Output a table of asset names, file names, and whether each file
        exists in the root folder."""

        output = []
        asset_list = ASSETS.copy()
        for asset_name, asset_fname in asset_list.items():
            fpath = self.asset_path(asset_name)
            output.append({
                'Asset name': asset_name,
                'Asset path': asset_fname,
                'Exists': True if os.path.exists(fpath) else False}
            )
        df = pd.DataFrame(output)
        display_or_print(df)

    # === DATA GENERATION ===

    # ====================
    def load_data(self, data: List[str]):
        """Convert provided data into form required for model training.

        Preprocess gold standard strings provided to raw inputs based
        on the features specified for restoration, then tokenize and
        convert to numpy format. Save the various assets in the model
        root folder.

        Required arguments:
        -------------------
        data: List[str]             A list of gold standard sentences
                                    (i.e. fully formatted sentences like
                                    "This is a sentence.")
        """

        self.generate_raw(data)
        show_ram_used()
        print()
        self.tokenize_inputs()
        show_ram_used()
        print()
        self.tokenize_outputs()
        show_ram_used()
        print()
        self.convert_inputs_to_numpy()
        show_ram_used()
        print()
        self.convert_outputs_to_numpy()
        show_ram_used()
        print()

    # ====================
    def generate_raw(self, data: List[str]):
        """Generate lists of raw X and y values to use as samples from
        datapoints (documents) in the data provided.

        Save in 'X_RAW' and 'Y_RAW' assets"""

        print(MESSAGE_GENERATING_RAW_SAMPLES)
        X = []
        y = []
        pbar = tqdm_(range(len(data)))
        for _ in pbar:
            pbar.set_postfix({
                'ram_usage': f"{psutil.virtual_memory().percent}%",
                'num_samples': len(X),
                'estimated_num_samples':
                    len(X) * (len(pbar) / (pbar.n + 1))
            })
            Xy = self.datapoint_to_Xy(data.pop(0))
            if Xy is not None:
                X_, y_ = Xy
                X.extend(X_)
                y.extend(y_)
        self.save_asset(X, 'X_RAW')
        self.save_asset(y, 'Y_RAW')
        print(SAVED_RAW_SAMPLES.format(num_samples=len(X)))

    # ====================
    def datapoint_to_Xy(self, datapoint: str) -> list:
        """Given a datapoint (i.e. a document in the data provided), generate a
        lists of X and y values for training."""

        # TODO: Implement case where one_of_each=False
        if self.one_of_each is not True:
            raise ValueError(ERROR_ONE_OF_EACH_FALSE_NOT_IMPLEMENTED)
        chars_orig = list_gclust(datapoint)
        chars = []
        classes = []
        while len(chars_orig) > 0:
            this_char = chars_orig.pop(0)
            if this_char in self.feature_chars:
                try:
                    classes[-1].append(this_char)
                except IndexError:
                    # If there are feature chars at the start of the doc, just
                    # ignore them until the first non-feature char
                    continue
            else:
                if self.capitalisation is True and this_char.isupper():
                    chars.append(this_char.lower())
                    classes.append(['U'])
                else:
                    chars.append(this_char)
                    classes.append([])
        if self.capitalisation is True:
            order = ['U'] + self.feature_chars
        else:
            order = self.feature_chars
        classes = [
            ''.join([c for c in order if c in class_])
            for class_ in classes
        ]
        assert len(chars) == len(classes)
        # Sliding windows
        X = []
        y = []
        if self.spaces is True:
            new_word_idxs = [0] + \
                [i + 1 for i, class_ in enumerate(classes) if ' ' in class_]
            for i in new_word_idxs:
                if i + self.seq_length < len(chars):
                    X.append(chars[i:i+self.seq_length])
                    y.append(classes[i:i+self.seq_length])
                else:
                    break
        if self.spaces is False:
            i = 0
            while i + self.seq_length < len(chars):
                X.append(chars[i:i+self.seq_length])
                y.append(classes[i:i+self.seq_length])
                i += self.char_shift
        assert all([len(x_) == self.seq_length for x_ in X])
        assert all([len(y_) == self.seq_length for y_ in y])
        return X, y

    # ====================
    def tokenize_inputs(self):
        """Tokenize inputs (X)"""

        print(MESSAGE_TOKENIZING_INPUTS)
        self.tokenize('X_TOKENIZER', 'X_RAW', 'X_TOKENIZED')

    # ====================
    def tokenize_outputs(self):
        """Tokenize outputs (y)"""

        print(MESSAGE_TOKENIZING_OUTPUTS)
        self.tokenize('Y_TOKENIZER', 'Y_RAW', 'Y_TOKENIZED')

    # ====================
    def tokenize(self, tokenizer_name: str, raw_asset_name: str,
                 tokenized_asset_name: str, char_level: bool):
        """Open an asset, create and fit a Keras tokenizer, tokenize the
        asset, and save both the tokenizer and the tokenized data"""

        data = self.get_asset(raw_asset_name)
        tokenizer = Tokenizer(oov_token='OOV', filters='')
        tokenizer.fit_on_texts(data)
        tokenized = tokenizer.texts_to_sequences(data)
        self.save_asset(tokenized, tokenized_asset_name)
        print(SAVED_TOKENIZED_SAMPLES.format(
            num_samples=len(tokenized),
            tokenized_asset_name=tokenized_asset_name
        ))
        self.save_asset(tokenizer, tokenizer_name)
        print(SAVED_TOKENIZER.format(
            num_categories=self.get_num_categories(tokenizer_name),
            tokenizer_name=tokenizer_name
        ))

    # ====================
    def convert_inputs_to_numpy(self):
        """Convert inputs (X) to numpy format"""

        print(MESSAGE_CONVERTING_INPUTS_TO_NUMPY)
        self.pickle_to_numpy('X_TOKENIZED', 'X')

    # ====================
    def convert_outputs_to_numpy(self):
        """Convert outputs (y) to numpy format"""

        print(MESSAGE_CONVERTING_OUTPUTS_TO_NUMPY)
        self.pickle_to_numpy('Y_TOKENIZED', 'Y')

    # ====================
    def pickle_to_numpy(self, pickle_asset_name: str, numpy_asset_name: str):
        """Open a .pickle asset, convert to a numpy array, and save as a .npy
        asset"""

        data_pickle = self.get_asset(pickle_asset_name)
        data_np = np.array(data_pickle)
        self.save_asset(data_np, numpy_asset_name)
        print(SAVED_NUMPY_ARRAY.format(
            shape=str(data_np.shape),
            numpy_asset_name=numpy_asset_name
        ))

    # ====================
    def preview_samples(self, k: int = 10):

        X = self.get_asset('X', mmap=True)
        y = self.get_asset('Y', mmap=True)
        all_idxs = range(len(X))
        idxs = sample(all_idxs, k)
        outputs = []
        for idx in idxs:
            restored = self.Xy_to_output(X[idx], y[idx])
            outputs.append((f"{idx:,}", restored))
        outputs_df = pd.DataFrame(outputs, columns=['Index', 'Output'])
        prev_colwidth = pd.options.display.max_colwidth
        pd.set_option('display.max_colwidth', None)
        display_or_print(outputs_df)
        pd.set_option('display.max_colwidth', prev_colwidth)

    # ====================
    def show_num_samples(self):

        X = self.get_asset('X', mmap=True)
        print(len(X))

    # === PREPROCESSING ===

    # ====================
    def preprocess_raw_str(self, raw_str):
        """Preprocess a raw string for input to a model"""

        if self.capitalisation is True:
            input_str = raw_str.lower()
        else:
            input_str = raw_str
        for fc in self.feature_chars:
            input_str = input_str.replace(fc, '')
        return input_str

    # ====================
    def input_str_to_model_input(self, input_str):
        """Prepare a raw string for input to a model"""

        input_str = list_gclust(input_str)
        tokenized = self.tokenize_input_str(input_str)
        encoded = self.encode_tokenized_str(tokenized)
        return encoded

    # ====================
    def tokenize_input_str(self, input_str):
        """Tokenize an input string"""

        input_len = len_gclust(input_str)
        if input_len > self.seq_length:
            error_msg = ERROR_INPUT_STR_TOO_LONG.format(
                seq_len=self.seq_length,
                len_input=input_len
            )
            raise ValueError(error_msg)
        # input_str = self.impose_seq_length(input_str)
        tokenizer = self.get_asset('X_TOKENIZER')
        tokenized = tokenizer.texts_to_sequences([input_str])
        tokenized = pad_sequences(
            tokenized, maxlen=self.seq_length, padding='post'
        )
        return tokenized

    # ====================
    def encode_tokenized_str(self, tokenized: str):

        num_X_categories = self.get_num_categories('X_TOKENIZER')
        encoded = to_categorical(tokenized, num_X_categories)
        return encoded

    # === PREDICTION & PREVIEW

    # ====================
    def Xy_to_output(self, X: list, y: list) -> str:
        """Generate a raw string (text with features) from model input (X)
        and output (y)"""

        X_tokenizer = self.get_asset('X_TOKENIZER')
        X_decoded = X_tokenizer.sequences_to_texts([X])[0].split()
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        assert len(X_decoded) == len(y_decoded)
        output_parts = [self.char_and_class_to_output_str(X_, y_)
                        for X_, y_ in zip(X_decoded, y_decoded)]
        output = ''.join(output_parts)
        return output

    # ====================
    def get_num_categories(self, tokenizers: Str_or_List) -> Int_or_Tuple:
        """Get the number of categories in one or more tokenizers.

        If a single tokenizer name is passed, the return value is an integer.
        If a list of tokenizer names is passed, the return value is a tuple of
        integers."""

        tokenizers = str_or_list_to_list(tokenizers)
        tokenizers = [self.get_asset(t) for t in tokenizers]
        num_categories = tuple([len(t.word_index) + 1 for t in tokenizers])
        return only_or_all(num_categories)

    # ====================
    def predict(self, raw_str: str):
        """Get the predicted output for a string of length less than or
        equal to the model sequence length"""

        input_str = self.preprocess_raw_str(raw_str)
        input_str = list_gclust(input_str)
        X_encoded = self.input_str_to_model_input(raw_str)
        predicted = self.model.model.predict(X_encoded)
        y = np.argmax(predicted, axis=2)[0]
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        output_parts = [self.char_and_class_to_output_str(X_, y_)
                        for X_, y_ in zip(input_str, y_decoded)]
        output = ''.join(output_parts)
        return output

    # ====================
    def predict_docs(self, docs: Str_or_List_or_Series) -> Str_or_List:
        """Get the predicted output for a single doc, or a list
        or pandas Series of docs.

        Required arguments
        ------------------
        docs:                       The documents to restore features to.
            Str_or_List_or_Series   Documents are preprocessed prior to
                                    prediction, so docs can contain either
                                    raw character sequences or gold standard
                                    formatted texts.
        """

        docs = str_or_list_or_series_to_list(docs)
        outputs = []
        pbar = tqdm_(range(len(docs)))
        for i in pbar:
            pbar.set_postfix(
                {'ram_usage': f"{psutil.virtual_memory().percent}%"})
            outputs.append(self.predict_single_doc(docs[i]))
        return only_or_all(outputs)

    # ====================
    def predict_single_doc(self, raw_str: str) -> str:
        """Get the predicted output for a document (any length)."""

        input_str = self.preprocess_raw_str(raw_str)
        if self.spaces is True:
            output = self.predict_doc_spaces_true(input_str)
        else:
            output = self.predict_doc_spaces_false(input_str)
        return output

    # ====================
    def predict_doc_spaces_true(self,
                                input_str: str) -> str:

        all_output = []
        prefix = ''
        while input_str:
            restore_until = self.seq_length - len_gclust(prefix)
            text_to_restore = \
                prefix + ''.join(list_gclust(input_str)[:restore_until])
            input_str = \
                ''.join(list_gclust(input_str)[restore_until:])
            chunk_restored = self.predict(text_to_restore)
            chunk_restored = chunk_restored.split(' ')
            prefix = self.preprocess_raw_str(
                ''.join(chunk_restored[-CHUNKER_NUM_PREFIX_WORDS:])
            )
            all_output.extend(chunk_restored[:-CHUNKER_NUM_PREFIX_WORDS])
        output = ' '.join(all_output)
        # Add any text remaining in 'prefix'
        if prefix:
            prefix_restored = self.predict(prefix)
            output = output + ' ' + prefix_restored.strip()
        return output

    # ====================
    def predict_doc_spaces_false(self, input_str: str) -> str:

        all_output = []
        prefix = ''
        while input_str:
            restore_until = self.seq_length - len(prefix)
            text_to_restore = prefix + input_str[:restore_until]
            input_str = input_str[restore_until:]
            chunk_restored = self.predict(text_to_restore)
            prefix = ''.join(chunk_restored[-CHUNKER_NUM_PREFIX_CHARS:])
            all_output.extend(chunk_restored[:-CHUNKER_NUM_PREFIX_CHARS])
        output = ''.join(all_output)
        # Add any text remaining in 'prefix'
        if prefix:
            output = output + self.predict(prefix).strip()
        return output

    # === GRID SEARCH ===

    # ====================
    def add_grid_search(self,
                        grid_search_name: str,
                        units: Union[int, list],
                        batch_size: Union[int, list],
                        dropout: Union[float, list],
                        recur_dropout: Union[float, list],
                        keep_size: float,
                        val_size: float,
                        epochs: int):

        attrs = locals()
        del attrs['self']
        self.grid_search = BiLSTMCharFeatureRestorerGridSearch(self, **attrs)

    # ====================
    def load_grid_search(self, grid_search_name: str):

        self.grid_search = \
            BiLSTMCharFeatureRestorerGridSearch.load(self, grid_search_name)

    # ====================
    def grid_search_path(self):

        grid_search_path_ = \
            os.path.join(self.root_folder, GRID_SEARCH_PATH_NAME)
        mk_dir_if_does_not_exist(grid_search_path_)
        return grid_search_path_

    # === STATIC METHODS ===

    # ====================
    @staticmethod
    def decode_class_list(tokenizer, encoded: list) -> list:

        index_word = json.loads(tokenizer.get_config()['index_word'])
        decoded = [index_word[str(x)] for x in encoded]
        return decoded

    # ====================
    @staticmethod
    def char_and_class_to_output_str(X_: str, y_: str) -> str:

        if len(y_) > 0 and y_[0] == 'u':
            X_ = X_.upper()
            y_ = y_[1:]
        return X_ + y_
