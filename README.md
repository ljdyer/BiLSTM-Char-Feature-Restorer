# BiLSTM Char Feature Restorer

A Python library for training character-level BiLSTM models for restoration of features such as spaces, punctuation, and capitalization to unformatted texts.

E.g.
`thisisasentence -> This is a sentence.`

Developed and used for the paper "Comparison of Token- and Character-Level Approaches to Restoration of Spaces, Punctuation, and Capitalization in Various Languages", which is scheduled for publication in December 2022.

## Interactive demo

The quickest and best way to get acquainted with the library is through the interactive demo [here](https://colab.research.google.com/drive/1aS6_-5rX4TOaa-qHIBCSW07-xS7nihk4?usp=sharing), where you can walk through the steps involved in using the library and train a model for restoration of spaces, punctuation, and capitalization model using sample data from the Ted Talks dataset used in the paper.

Alternatively, scroll down for instructions on getting started and basic documentation.

## Getting started

### Install the library using `pip`

```
!pip install git+https://github.com/ljdyer/BiLSTM-Char-Feature-Restorer.git
```

### Import the `BiLSTMCharFeatureRestorer` class

```python
from bilstm_char_feature_restorer import BiLSTMCharFeatureRestorer
```

## Model training and feature restoration using the `BiLSTMFeatureRestorer` class

Multiple models can be trained on a single class instance. A single instance is used to train models using the same training data to restore the same set of features, so in our paper we used separate instances for each of **TedTalks**, **Brown**, **OshieteQA**, and **GujaratiNews**.

### Initialize a class instance

#### `BiLSTMCharFeatureRestorer.__init__`

```python
    # ====================
    def __init__(self,
                 root_folder: str,
                 capitalization: bool,
                 spaces: bool,
                 other_features: list,
                 seq_length: int,
                 one_of_each: bool = True,
                 char_shift: Optional[int] = None):
        """Initialize and train an instance of the class.

        Args:
          root_folder (str):
            The path to a folder to which to save model assets. The folder
            should not exist yet. It will be created.
          capitalization (bool):
            Whether or not models trained on the instance will attempt to
            restore capitalization (e.g. convert 'new york' to 'New York').
          spaces (bool):
            Whether or not models trained on the instance will attempt to
            restore spaces. (e.g. convert 'goodmorning' to 'good morning')
          other_features (list):
            A list of other characters that models trained on the instance
            will restore.
            E.g. to restore commas and periods, set other_features=['.', ','].
            The order in which the characters appear in output texts if a
            single character possesses more than one feature will be the
            same as the order in which they appear in this list.
            Spaces will appear after features in other_features.
          seq_length (int):
            The length in characters of model input sequences used for
            preprocessing, training, and prediction.
          one_of_each (bool, optional):
            If set to True, the model will only restore a maximum of one
            of each feature per character. E.g. '...' will never appear in
            output texts.
            If set to False, the model will restore an arbitrary number of
            each feature if it deems that more than one of a feature is
            appropriate. (Not implemented yet at the time of writing.)
            Defaults to True.
          char_shift (Optional[int], optional):
            Only required if spaces=False. The step size in characters for
            the sliding window when generating input data. Smaller values
            of char_shift generate larger numbers of training examples.
            Defaults to None.

        Raises:
          ValueError:
            If a folder already exists at root_folder.
          ValueError:
            If spaces=False but char_shift has not been provided.
        """
```

#### Example usage:

```python
restorer = BiLSTMCharFeatureRestorer(
    root_folder='drive/MyDrive/BiLSTMCharFeatureRestorer_demo',
    capitalization=True,
    spaces=True,
    other_features=[',', '.'],
    seq_length=200
)
```

<img src="readme-img/01-init.PNG"></img>

To be continued...