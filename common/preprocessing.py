# Sebastian Thomas (datascience at sebastianthomas dot de)

# type hints
from typing import Union

# operating system
from os import PathLike

# paths
from pathlib import PurePath

# data
import numpy as np

# signal processing
import librosa

# machine learning
import tensorflow as tf

# custom modules
from common.constants import COMMANDS, CATEGORIES, UNKNOWN_CATEGORY, \
    NUM_SAMPLES


def to_mfcc(file_path: Union[str, PathLike], n_mfcc: int = 13,
            hop_length: int = 512, n_fft: int = 2048) -> tf.Tensor:
    """Returns the mel frequency cepstral coefficients of the sound file at the
    given path, using the given parameters."""
    signal, _ = librosa.load(file_path, sr=None)
    padded_signal = np.concatenate((signal,
                                    np.zeros(NUM_SAMPLES - len(signal))))
    mfcc = librosa.feature.mfcc(padded_signal, sr=NUM_SAMPLES, n_mfcc=n_mfcc,
                                hop_length=hop_length, n_fft=n_fft).T

    return tf.convert_to_tensor(mfcc, dtype=tf.float64)


def to_features(file_path: Union[str, PathLike], n_mfcc: int = 13,
                hop_length: int = 512, n_fft: int = 2048) -> tf.Tensor:
    """Returns the features of the sound file at the given path."""
    # in order to feed the features into a convolutional neural network,
    # we increase the tensor rank by 1 (channel entry)
    return tf.expand_dims(to_mfcc(file_path, n_mfcc=n_mfcc,
                                  hop_length=hop_length, n_fft=n_fft), -1)


def to_label(file_path: PurePath) -> tf.Tensor:
    """Returns the label of the sound file at the given path."""
    command = file_path.parent.name
    category = command if command in COMMANDS else UNKNOWN_CATEGORY
    label = CATEGORIES.index(category)

    return tf.convert_to_tensor(label, dtype=tf.int8)
