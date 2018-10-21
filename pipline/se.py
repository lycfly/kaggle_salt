from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute,Conv2D,Add
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def sse_block(input):
    conv = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal",
                  activation='sigmoid', strides=(1, 1))(input)
    conv = multiply([input, conv])
    return conv

def scSE_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = squeeze_excite_block(input = x)
    sse = sse_block(input = x)
    x = Add(name=prefix)([cse, sse])

    return x