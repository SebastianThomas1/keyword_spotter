# Sebastian Thomas (datascience at sebastianthomas dot de)


from __future__ import annotations


# type hints
from typing import Union, NoReturn, Tuple, List

# base classes
from collections.abc import Callable

# machine learning
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, \
    MaxPool1D, MaxPool2D
from tensorflow.keras.regularizers import Regularizer


__all__ = ['Conv1DBlock', 'Conv2DBlock']


class Conv1DBlock(Model):
    """A convolutional block consisting of a convolutional layer of tensor
    rank 1, a batch normalization layer and a max pooling layer."""

    def __init__(self, filters: int,
                 kernel_size: Union[int, Tuple[int, int], List[int]] = 3,
                 padding: str = 'valid',
                 activation: Union[Callable, str] = 'relu',
                 kernel_regularizer: Union[Regularizer, type(None)] = None,
                 pool_size: Union[int, Tuple[int, int]] = 2,
                 strides: Union[int, Tuple[int, int], type(None)] = None,
                 name: Union[str, type(None)] = None) -> NoReturn:
        """Initializes this instance."""
        super().__init__(name=name)

        self._conv = Conv1D(filters, kernel_size, padding=padding,
                            activation=activation,
                            kernel_regularizer=kernel_regularizer,
                            name=self.name + '__conv')
        self._batch_normalization = BatchNormalization(
            name=self.name + '__batch_normalization')
        self._max_pool = MaxPool1D(pool_size, strides=strides,
                                   name=self.name + '__max_pool')

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """Returns the result of a forward pass through this instance."""
        return self._max_pool(self._batch_normalization(self._conv(inputs)))

    def get_config(self) -> dict:
        """Returns config data in form of a JSON-serializable dictionary."""
        conv_layer_config = self._conv.get_config()
        return {'filters': self._conv.filters,
                'kernel_size': self._conv.kernel_size,
                'padding': self._conv.padding,
                'activation': conv_layer_config['activation'],
                'kernel_regularizer': conv_layer_config['kernel_regularizer'],
                'pool_size': self._max_pool.pool_size,
                'strides': self._max_pool.strides,
                'name': self.name}

    @classmethod
    def from_config(cls, config: dict) -> Conv1DBlock:
        """Constructs instance from config data."""
        return cls(**config)


class Conv2DBlock(Model):
    """A convolutional block consisting of a convolutional layer of tensor
    rank 2, a batch normalization layer and a max pooling layer."""

    def __init__(self, filters: int,
                 kernel_size: Union[int, Tuple[int, int], List[int]] = 3,
                 padding: str = 'valid',
                 activation: Union[Callable, str] = 'relu',
                 kernel_regularizer: Union[Regularizer, type(None)] = None,
                 pool_size: Union[int, Tuple[int, int]] = 2,
                 strides: Union[int, Tuple[int, int], type(None)] = None,
                 name: Union[str, type(None)] = None) -> NoReturn:
        """Initializes this instance."""
        super().__init__(name=name)

        self._conv = Conv2D(filters, kernel_size, padding=padding,
                            activation=activation,
                            kernel_regularizer=kernel_regularizer,
                            name=self.name + '__conv')
        self._batch_normalization = BatchNormalization(
            name=self.name + '__batch_normalization')
        self._max_pool = MaxPool2D(pool_size, strides=strides,
                                   name=self.name + '__max_pool')

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """Returns the result of a forward pass through this instance."""
        return self._max_pool(self._batch_normalization(self._conv(inputs)))

    def get_config(self) -> dict:
        """Returns config data in form of a JSON-serializable dictionary."""
        conv_layer_config = self._conv.get_config()
        return {'filters': self._conv.filters,
                'kernel_size': self._conv.kernel_size,
                'padding': self._conv.padding,
                'activation': conv_layer_config['activation'],
                'kernel_regularizer': conv_layer_config['kernel_regularizer'],
                'pool_size': self._max_pool.pool_size,
                'strides': self._max_pool.strides,
                'name': self.name}

    @classmethod
    def from_config(cls, config: dict) -> Conv2DBlock:
        """Constructs instance from config data."""
        return cls(**config)
