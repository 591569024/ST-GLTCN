# -*- coding=utf-8 -*-

from keras.layers import  Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import (
Input, Activation, merge, Dense, Reshape, Embedding, Flatten, Dropout, Lambda, LSTM
)
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import os
import keras
from GLSTModel import Attention


def stdn(args):

    att_lstm_num, att_lstm_seq_len, lstm_seq_len, neighbor_size, external_dim = \
        args['att_lstm_num'], args['att_lstm_seq_len'], args['lstm_seq_len'], \
        args['neighbor_size'], args['external_dim']

    slide_len = neighbor_size * 2 + 1

    flatten_att_nbhd_inputs = [Input(shape=(2, slide_len, slide_len), name="att_nbhd_volume_input_time_{0}_{1}".format(att + 1, ts + 1)) for
                               ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]

    flatten_att_flow_inputs = [Input(shape=(2, slide_len, slide_len), name="att_flow_volume_input_time_{0}_{1}".format(att + 1, ts + 1)) for
                               ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]

    # take out the corresponding local and flow data for attention
    att_nbhd_inputs = []
    att_flow_inputs = []
    for att in range(att_lstm_num):
        att_nbhd_inputs.append(flatten_att_nbhd_inputs[att * att_lstm_seq_len:(att + 1) * att_lstm_seq_len])
        att_flow_inputs.append(flatten_att_flow_inputs[att * att_lstm_seq_len:(att + 1) * att_lstm_seq_len])

    # external data for attention, holiday and weather
    att_lstm_inputs = [Input(shape=(att_lstm_seq_len, external_dim), name="att_lstm_input_{0}".format(att + 1)) for
                       att in range(att_lstm_num)]

    # local, flow, external data for local cnn
    nbhd_inputs = [Input(shape=(2, slide_len, slide_len), name="nbhd_volume_input_time_{0}".format(ts + 1)) for
                   ts in range(lstm_seq_len)] # include many sequence
    flow_inputs = [Input(shape=(2, slide_len, slide_len), name="flow_volume_input_time_{0}".format(ts + 1)) for
                   ts in range(lstm_seq_len)]
    lstm_inputs = Input(shape=(lstm_seq_len, external_dim), name="lstm_input")

    # short-term part, repeat three times
    # conv (local + flow) data --> Multiply fuse
    nbhd_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time0_{0}".format(ts + 1))(nbhd_inputs[ts]) for
                  ts in range(lstm_seq_len)] # Conv for different interval in current
    nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time0_{0}".format(ts + 1))(nbhd_convs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="flow_convs_time0_{0}".format(ts + 1))(flow_inputs[ts]) for
                  ts in range(lstm_seq_len)] # Conv for past l(2) interval for flow gate
    flow_convs = [Activation("relu", name="flow_convs_activation_time0_{0}".format(ts + 1))(flow_convs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_gates = [Activation("sigmoid", name="flow_gate0_{0}".format(ts + 1))(flow_convs[ts]) for
                  ts in range(lstm_seq_len)] # use sigmoid to scale the range of flow conv into (0, 1)
    nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for
                  ts in range(lstm_seq_len)] # Multiply local and flow

    # 2nd level gate
    nbhd_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time1_{0}".format(ts + 1))(nbhd_convs[ts]) for
                  ts in range(lstm_seq_len)] # ! here we use nbhd_convs(the result of first level)
    nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time1_{0}".format(ts + 1))(nbhd_convs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="flow_convs_time1_{0}".format(ts + 1))(flow_inputs[ts]) for
                  ts in range(lstm_seq_len)] # flow input contai
    flow_convs = [Activation("relu", name="flow_convs_activation_time1_{0}".format(ts + 1))(flow_convs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_gates = [Activation("sigmoid", name="flow_gate1_{0}".format(ts + 1))(flow_convs[ts]) for
                  ts in range(lstm_seq_len)]
    nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]

    # 3rd level gate
    nbhd_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time2_{0}".format(ts + 1))(nbhd_convs[ts]) for
                  ts in range(lstm_seq_len)]
    nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time2_{0}".format(ts + 1))(nbhd_convs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="flow_convs_time2_{0}".format(ts + 1))(flow_inputs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_convs = [Activation("relu", name="flow_convs_activation_time2_{0}".format(ts + 1))(flow_convs[ts]) for
                  ts in range(lstm_seq_len)]
    flow_gates = [Activation("sigmoid", name="flow_gate2_{0}".format(ts + 1))(flow_convs[ts]) for
                  ts in range(lstm_seq_len)]
    nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]

    # dense part
    nbhd_vecs = [Flatten(name="nbhd_flatten_time_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
    nbhd_vecs = [Dense(units=128, name="nbhd_dense_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for
                 ts in range(lstm_seq_len)]
    nbhd_vecs = [Activation("relu", name="nbhd_dense_activation_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for
                 ts in range(lstm_seq_len)]

    # feature concatenate
    nbhd_vec = merge.Concatenate(axis=-1)(nbhd_vecs)
    nbhd_vec = Reshape(target_shape=(lstm_seq_len, 128))(nbhd_vec)
    lstm_input = merge.Concatenate(axis=-1)([lstm_inputs, nbhd_vec]) # lstm_input: external data, nbhd_vec: cnn flatten result

    # lstm: the result of current interval
    lstm = LSTM(units=128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

    # attention part, repeat 3 times
    # # conv (local + flow) data --> Multiply fuse
    att_nbhd_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="att_nbhd_convs_time0_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_inputs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)] # att local conv
    att_nbhd_convs = [[Activation("relu", name="att_nbhd_convs_activation_time0_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="att_flow_convs_time0_{0}_{1}".format(att + 1, ts + 1))(att_flow_inputs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)] # att flow
    att_flow_convs = [[Activation("relu", name="att_flow_convs_activation_time0_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_gates = [[Activation("sigmoid", name="att_flow_gate0_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)] # att flow gate
    att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)] # multiply to fuse local and flow gate


    att_nbhd_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="att_nbhd_convs_time1_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)] # ! note the input
    att_nbhd_convs = [[Activation("relu", name="att_nbhd_convs_activation_time1_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="att_flow_convs_time1_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_convs = [[Activation("relu", name="att_flow_convs_activation_time1_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_gates = [[Activation("sigmoid", name="att_flow_gate1_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]


    att_nbhd_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="att_nbhd_convs_time2_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_nbhd_convs = [[Activation("relu", name="att_nbhd_convs_activation_time2_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_convs = [[Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="att_flow_convs_time2_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_convs = [[Activation("relu", name="att_flow_convs_activation_time2_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_flow_gates = [[Activation("sigmoid", name="att_flow_gate2_{0}_{1}".format(att + 1, ts + 1))(att_flow_convs[att][ts]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for
                       ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

    # reshape cnn into (att_lstm_seq_len, 128),
    # then concatenate with lstm_input(external data)--> every day data
    att_nbhd_vecs = [[Flatten(name="att_nbhd_flatten_time_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_convs[att][ts]) for
                      ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_nbhd_vecs = [[Dense(units=128, name="att_nbhd_dense_time_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_vecs[att][ts]) for
                      ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
    att_nbhd_vecs = [[Activation("relu", name="att_nbhd_dense_activation_time_{0}_{1}".format(att + 1, ts + 1))(att_nbhd_vecs[att][ts]) for
                      ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

    att_nbhd_vec = [merge.Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(att_lstm_num)]
    att_nbhd_vec = [Reshape(target_shape=(att_lstm_seq_len, 128))(att_nbhd_vec[att]) for att in range(att_lstm_num)]
    att_lstm_input = [merge.Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in range(att_lstm_num)]

    # attention part : every day data --> lstm
    att_lstms = [LSTM(units=128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for
                 att in range(att_lstm_num)]

    print('Before')
    # compare
    att_low_level = [Attention()([att_lstms[att], lstm]) for att in range(att_lstm_num)]
    att_low_level = merge.Concatenate(axis=-1)(att_low_level)
    att_low_level = Reshape(target_shape=(att_lstm_num, 128))(att_low_level)

    print('After')

    att_high_level = LSTM(units=128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

    lstm_all = merge.Concatenate(axis=-1)([att_high_level, lstm])
    lstm_all = Dense(units=2 * slide_len * slide_len)(lstm_all)
    lstm_all = Reshape((2, slide_len, slide_len))(lstm_all)
    pred_volume = Activation('tanh')(lstm_all)

    # every input is a list, only be this way
    inputs = \
        flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs]
    # flatten_att_nbhd_inputs: (att_lstm_num * att_lstm_seq_len, 2, slide_len, slide_len)
    # flatten_att_flow_inputs: (att_lstm_num * att_lstm_seq_len, 2, slide_len, slide_len)
    # att_lstm_inputs: (att_lstm_num, att_lstm_seq_len, 18)
    # nbhd_inputs: (lstm_seq_len, 2, slide_len, slide_len)
    # flow_inputs: (lstm_seq_len, 2, slide_len, slide_len)
    # lstm_inputs: (lstm_seq_len, 18)

    print('flatten_att_nbhd_inputs:', att_lstm_num * att_lstm_seq_len, 2, slide_len, slide_len)
    print('flatten_att_flow_inputs:', att_lstm_num * att_lstm_seq_len, 2, slide_len, slide_len)
    print('att_lstm_inputs:', att_lstm_num, att_lstm_seq_len, 18)
    print('nbhd_inputs:', lstm_seq_len, 2, slide_len, slide_len)
    print('flow_inputs:', lstm_seq_len, 2, slide_len, slide_len)
    print('lstm_inputs:', lstm_seq_len, 18)

    model = Model(inputs=inputs, outputs=pred_volume)
    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #model = stdn(3, 7, 7)
    #model.summary()
