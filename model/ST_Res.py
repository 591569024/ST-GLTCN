# -*- coding=utf-8 -*-

"""
ResModel的keras版本
"""
from keras.layers import  Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import (
Input, Activation, merge, Dense, Reshape, Embedding, Flatten, Dropout, Lambda
)
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import os
from model.GLSTModel import ilayer, rest_unit

def resnet(args):

    neighbor_size, len_close, len_period, len_trend, nb_residual_unit, external_dim = \
        args['neighbor_size'], args['len_close'], args['len_period'], \
        args['len_trend'], args['nb_residual_unit'], args['external_dim']
    external = True
    is_BN = True
    is_fuse = True

    slide_length = neighbor_size * 2 + 1

    c_conf = (len_close, 2, slide_length, slide_length)
    p_conf = (len_period, 2, slide_length, slide_length)
    t_conf = (len_trend, 2, slide_length, slide_length)

    main_input = []
    main_output = []

    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:

            len_seq, nb_flow, height, width = conf
            input = Input(shape=(nb_flow * len_seq, height, width))
            main_input.append(input)
            # conv1
            # output = (image - filter + 2 * padding) / stride + 1
            conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(input)
            #res_ouput
            res_output = rest_unit(is_BN, nb_residual_unit)(conv1)
            # conv2
            activation = Activation('relu')(res_output)
            conv2 = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)
            main_output.append(conv2)

    # hamadard
    if is_fuse:
        new_output = []
        for output in main_output:
            new_output.append(ilayer()(output))
        main_output = merge.Add()(new_output)
    else:
        main_output = merge.Add()(main_output)

    if external:
        external_input = Input(shape=(external_dim,))
        main_input.append(external_input)
        embedding = Dense(units = 10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=2 * slide_length * slide_length)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((2, slide_length, slide_length))(activation)

        main_output = merge.Add()([main_output, external_output])

    main_output = Activation('tanh')(main_output)
    model = Model(inputs=main_input, outputs=main_output)

    return model

if __name__ == '__main__':
    model = resnet()
    model.summary()