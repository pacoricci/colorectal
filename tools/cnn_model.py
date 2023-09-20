from keras import layers, Input, Model, regularizers
from tools import constants, custom_layers


def gen_model():
    """
    The core of the model
    :return:
    """
    _input_shape = Input(shape=[150, 150, 3])
    _out = layers.GaussianNoise(0.03)(_input_shape)
    _out = layers.Conv2D(32,
                         (8, 8),
                         strides=(4, 4),
                         padding='valid',
                         use_bias=False,
                         kernel_regularizer=regularizers.L2(constants.WEIGHT_DECAY))(_out)
    _out = layers.BatchNormalization(momentum=constants.BATCH_NORMALIZATION_MOMENTUM)(_out)
    _out = layers.ReLU()(_out)
    _out = layers.MaxPooling2D((2, 2))(_out)
    _out = custom_layers.se_block([18, 18, 32])(_out)
    _out = custom_layers.cat3([18, 18, 32], 64)(_out)
    _out = layers.MaxPooling2D((2, 2))(_out)
    _out = custom_layers.se_block([9, 9, 64])(_out)
    _out = custom_layers.cat3([9, 9, 64], 128)(_out)
    _out = layers.GlobalAveragePooling2D()(_out)
    _out = layers.Dense(32, activation='relu')(_out)
    _out1 = layers.Dense(8,
                         activation='softmax',
                         name='cce',
                         kernel_regularizer=regularizers.L2(constants.WEIGHT_DECAY))(_out)
    out2 = custom_layers.Ident(name='ole')(_out)
    return Model(_input_shape, [_out1, out2])
