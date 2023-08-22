from keras import layers, Input, Model, regularizers
from colorectal import constants


def se_block(shape, ratio=4):
    _ch = shape[-1]
    _input_shape = Input(shape, batch_size=constants.BATCH_SIZE)
    _y = layers.GlobalAveragePooling2D()(_input_shape)
    _y = layers.Dense(_ch // ratio, activation='relu', kernel_regularizer=regularizers.L2(constants.WEIGHT_DECAY))(_y)
    _y = layers.Dense(_ch, activation='sigmoid', kernel_regularizer=regularizers.L2(constants.WEIGHT_DECAY))(_y)
    _y = layers.Multiply()([_input_shape, _y])
    return Model(_input_shape, _y)


def cat3(shape, out_channels):
    _input_shape = Input(shape, batch_size=constants.BATCH_SIZE)
    _tower_1 = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same')(_input_shape)
    _tower_1 = layers.Conv2D(out_channels // 4,
                             (1, 1),
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=regularizers.L2(constants.WEIGHT_DECAY))(_tower_1)
    _tower_2 = layers.Conv2D(out_channels // 4, (3, 3),
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=regularizers.L2(constants.WEIGHT_DECAY))(_input_shape)
    _tower_3 = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(_input_shape)
    _merged = layers.Concatenate(axis=-1)([_tower_1, _tower_2, _tower_3])
    _merged = layers.BatchNormalization(momentum=constants.BATCH_NORMALIZATION_MOMENTUM)(_merged)
    _merged = layers.ReLU()(_merged)
    return Model(_input_shape, _merged)


class Ident(layers.Layer):
    def call(self, inputs):
        return inputs
