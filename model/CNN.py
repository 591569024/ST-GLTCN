# -*- coding=utf-8 -*-
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import (
    Input, Activation, merge, Dense, Reshape, Embedding, Flatten, Dropout, Lambda, LSTM, ConvLSTM2D
)
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import os

class ilayer(Layer):
    """
    Hadamard
    """

    def __init__(self):
        super(ilayer, self).__init__()

    def build(self, input_shape):
        """
        define the weight and bias here
        :param input_shape:
        :return:
        """
        initial_weight = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight)
        self.trainable_weights = [self.W]

    def call(self, inputs, **kwargs):
        # define the logic for processing the network
        return inputs * self.W

    def compute_output_shape(self, input_shape):
        # define the logic for changing the shape
        return input_shape

def cnn_unit(is_BN, nb_cnn):
    """
    used to deal with current local data
    :param is_BN:
    :param nb_cnn: the number of cnn
    :return:
    """

    def func(input):
        for i in range(nb_cnn):
            if is_BN:
                input = BatchNormalization(axis=1)(input)
            activation = Activation('relu')(input)
            input = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(activation)
        return input

    return func

def cnn(args):
    len_local = args['len_local']
    neighbor_size = args['neighbor_size']
    is_BN = args['is_BN']
    nb_cnn = args['nb_cnn']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # todo consider the model here
    cnn_out = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_stacked_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    main_output = Activation('tanh')(feature_stacked_local_flow)

    model = Model(inputs=main_input, outputs=main_output, name='CNN')
    return model

if __name__ == '__main__':
    pass