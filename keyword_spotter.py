# Sebastian Thomas (datascience at sebastianthomas dot de)

from __future__ import annotations

# type hints
from typing import ClassVar, Union, Optional, NoReturn, BinaryIO

# operation system
from os import PathLike

# machine learning
import tensorflow as tf
from tensorflow.keras.models import load_model

# custom modules
from common.constants import CATEGORIES, CLASSIFIER_PATH
from common.preprocessing import to_features


__all__ = ['KeywordSpotter', 'KeywordSpotterType']


class KeywordSpotterType:
    """Type of KeywordSpotter singleton."""

    _instance: ClassVar[Optional[KeywordSpotterType]] = None

    def __new__(cls) -> NoReturn:
        """Constructs new instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> NoReturn:
        """Initializes instance."""
        try:
            self._classifier = load_model(CLASSIFIER_PATH)
        except OSError:
            self._classifier = None

    def update(self) -> NoReturn:
        """Updates instance by reloading the classifier field."""
        self.__init__()

    def predict(self, file: Union[str, PathLike, BinaryIO]) -> str:
        """Predicts keyword spoken in file.

        Raises FileNotFoundError if no classifier is saved at
        CLASSIFIER_PATH."""

        try:
            n_mfcc = self._classifier.input_shape[2]
        except AttributeError:
            try:
                self._classifier = load_model(CLASSIFIER_PATH)
            except OSError:
                raise FileNotFoundError('No classifier could be found.')
            else:
                n_mfcc = self._classifier.input_shape[2]

        features = to_features(file, n_mfcc=n_mfcc)
        logits = self._classifier(tf.expand_dims(features, 0))[0]

        return CATEGORIES[int(tf.argmax(logits))]


KeywordSpotter = KeywordSpotterType()
