# -*- coding=utf-8 -*-

from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import (
Input, Activation, merge, Dense, Reshape, Embedding, Flatten, Dropout, Lambda, LSTM, ConvLSTM2D, Concatenate
)
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import os

external_len = 10

class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # three weight
        # Wh: att_size, att_size -->  previous hidden state
        # Wq: query_dim, att_size --> target hidden state
        # V:  att_size, 1 --> tanh
        # score(previous, target) = Vt * tanh(Wh * previous + target * Wq + b???) --> (1, 1)

        # the dimension of previous hidden state
        self.att_size = input_shape[0][-1]
        # the dimension of target hidden state
        self.query_dim = input_shape[1][-1]

        self.Wq = self.add_weight(name='kernal_query_features', shape=(self.query_dim, self.att_size),
                                  initializer='glorot_normal', trainable=True)

        self.Wh = self.add_weight(name='kernal_hidden_features', shape=(self.att_size, self.att_size),
                                  initializer='glorot_normal', trainable=True)

        self.v = self.add_weight(name='query_vector', shape=(self.att_size, 1),
                                 initializer='zeros', trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        # score(previous, target) = Vt * tanh(Wh * memory + target * Wq)

        memory, query = inputs[0], inputs[1]
        hidden = K.dot(memory, self.Wh) + K.expand_dims(K.dot(query, self.Wq), 1)
        hidden = K.tanh(hidden)
        # remove the dimension whose shape is 1
        s = K.squeeze(K.dot(hidden, self.v), -1)

        # compute the weight use soft_max
        s = K.softmax(s)

        return K.sum(memory * K.expand_dims(s), axis=1)

    def compute_output_shape(self, input_shape):
        att_size = input_shape[0][-1]
        batch = input_shape[0][0]
        return batch, att_size

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

def _bn_relu_conv(is_BN=True):
    def f(input):
        if is_BN:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(64, (3, 3), strides=(1, 1), padding="same")(activation)
    return f

def _residual_unit(is_BN):
    def f(input):
        residual = _bn_relu_conv(is_BN)(input)
        residual = _bn_relu_conv(is_BN)(residual)
        return merge.Add()([residual, input])
    return f

def rest_unit(is_BN, nb_residual_unit):
    def func(input):
        for i in range(nb_residual_unit):
            input = _residual_unit(is_BN)(input)
        return input
    return func

def cnn_unit(is_BN, nb_cnn, nb_filter=32):
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
            input = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(activation)
        return input
    return func

def cnn_by_channel(stack_cnn, time_length, is_BN):

    def func(input):
        for i in range(stack_cnn):
            output = []
            for time in range(time_length):
                time_piece = Lambda(lambda x: x[:, time * 2:time * 2 + 2, :, :])(input)
                time_piece = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(time_piece)
                time_piece = Activation('relu')(time_piece)
                output.append(time_piece)

            # concatenate ask at least 2 elements
            if time_length > 1:
                input = merge.Concatenate(axis=1)(input)
            else:
                input = output[0]

            if i != stack_cnn - 1 and is_BN == True:
                input = BatchNormalization(axis=1)(input)
        return input
    return func

def glst_net(args):
    len_global, len_local, neighbor_size, is_BN, nb_res_unit, nb_cnn, dataset, width, height = \
        args['len_global'], args['len_local'], args['neighbor_size'], args['is_BN'], \
        args['nb_res_unit'], args['nb_cnn'], args['dataset'], args['width'], args['height']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

    # ==============================
    # deal with the global local external data
    if dataset == 'bj_taxi':
        g_vacation = Input(shape=(len_global,), dtype='int32', name='g_vacation')
        g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
        g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
        g_weather = Input(shape=(len_global,), dtype='int32', name='g_weather')
        g_continuous_external = Input(shape=(len_global, 2), dtype='float32', name='g_continuous_external')
        main_input.append(g_vacation)
        main_input.append(g_hour)
        main_input.append(g_dayOfWeek)
        main_input.append(g_weather)
        main_input.append(g_continuous_external)

        embed_g_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_global)(g_vacation)
        embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
        embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)
        embed_g_weather = Embedding(output_dim=2, input_dim=17, input_length=len_global)(g_weather)

        g_external = merge.Concatenate(axis=-1)(
            [embed_g_holiday, embed_g_hour, embed_g_dayOfWeek, embed_g_weather, g_continuous_external])
    else:
        g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
        g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
        main_input.append(g_hour)
        main_input.append(g_dayOfWeek)

        embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
        embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)

        g_external = merge.Concatenate(axis=-1)([embed_g_hour, embed_g_dayOfWeek])

    g_out = Flatten()(g_external)
    g_out = Dense(units=50)(g_out)
    g_out = Dropout(rate=0.1)(g_out)
    g_out = Activation('relu')(g_out)
    g_out = Dense(units=2 * width * height)(g_out)
    g_out = Activation('relu')(g_out)
    feature_external_g = Reshape((2, width, height))(g_out)


    # ==============================
    # deal with the local external data
    if dataset == 'bj_taxi':
        t_vacation = Input(shape=(len_local,), dtype='int32', name='t_vacation')
        t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
        t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
        t_weather = Input(shape=(len_local,), dtype='int32', name='t_weather')
        t_continuous_external = Input(shape=(len_local, 2), dtype='float32', name='t_continuous_external')
        main_input.append(t_vacation)
        main_input.append(t_hour)
        main_input.append(t_dayOfWeek)
        main_input.append(t_weather)
        main_input.append(t_continuous_external)

        embed_t_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_local)(t_vacation)
        embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
        embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)
        embed_t_weather = Embedding(output_dim=2, input_dim=17, input_length=len_local)(t_weather)

        t_external = merge.Concatenate(axis=-1)(
            [embed_t_holiday, embed_t_hour, embed_t_dayOfWeek, embed_t_weather, t_continuous_external])
    else:
        t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
        t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
        main_input.append(t_hour)
        main_input.append(t_dayOfWeek)

        embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
        embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)

        t_external = merge.Concatenate(axis=-1)([embed_t_hour, embed_t_dayOfWeek])

    t_out = Flatten()(t_external)
    t_out = Dense(units=50)(t_out)
    t_out = Dropout(rate=0.1)(t_out)
    t_out = Activation('relu')(t_out)
    t_out = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(t_out)
    t_out = Activation('relu')(t_out)
    feature_external_t = Reshape((2, neighbor_slide_len, neighbor_slide_len))(t_out)

    # ==============================
    # deal with the current local flow data
    current_local_flow = \
        Input(shape=(2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='current_local_flow')
    main_input.append(current_local_flow)

    cnn_out = cnn_unit(is_BN, nb_cnn)(current_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_current_local_flow)

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # todo consider the model here
    cnn_out = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_stacked_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_stacked_local_flow)

    # ==============================
    # fuse local stacked flow feature and current local flow feature, then fuse local external feature
    local_output = []
    for feature in local_feature_list:
        local_output.append(ilayer()(feature))
    feature_local_flow = merge.Add()(local_output)

    # todo consider the fuse method
    feature_local = merge.Add()([feature_external_t, feature_local_flow])

    # ==============================
    # deal with the global flow data, then fuse global external feature
    global_flow = \
        Input(shape=(len_global * 2, width, height), dtype='float32', name='global_flow')
    main_input.append(global_flow)

    # todo consider the model here
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(global_flow)
    activation = Activation('relu')(conv1)
    res_out = rest_unit(is_BN, nb_res_unit)(activation)
    activation = Activation('relu')(res_out)
    feature_global_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)


    # todo consider the fuse method
    feature_global = merge.Add()([feature_external_g, feature_global_flow])

    # reshape the global feaure
    feature_global = Flatten()(feature_global)
    feature_global = Dense(units=200)(feature_global)
    feature_global = Dropout(rate=0.1)(feature_global)
    feature_global = Dense(units= 2 * neighbor_slide_len * neighbor_slide_len)(feature_global)
    feature_global = Activation('relu')(feature_global)
    feature_global = Reshape((2, neighbor_slide_len, neighbor_slide_len))(feature_global)

    # ==============================
    # fuse local and global feature to predict
    # index_cut = Input(shape=(2,), dtype='int32', name='index_cut')
    # main_input.append(index_cut)
    # row, column = index_cut[0].eval(), index_cut[1].eval(session=K.get_session())

    # fuse the global and local feature
    output = []
    for feature in [feature_global, feature_local]:
        output.append(ilayer()(feature))
    main_output = merge.Add()(output)

    main_output = Activation('tanh')(main_output)

    model = Model(inputs=main_input, outputs=main_output, name='self')
    return model

def glst_net_no_global(args):
    len_local, neighbor_size, is_BN, nb_cnn = \
        args['len_local'], args['neighbor_size'], args['is_BN'], args['nb_cnn']
    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

    # ==============================
    # deal with the local external data
    t_vacation = Input(shape=(len_local,), dtype='int32', name='t_vacation')
    t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
    t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
    t_weather = Input(shape=(len_local,), dtype='int32', name='t_weather')
    t_continuous_external = Input(shape=(len_local, 2), dtype='float32', name='t_continuous_external')
    main_input.append(t_vacation)
    main_input.append(t_hour)
    main_input.append(t_dayOfWeek)
    main_input.append(t_weather)
    main_input.append(t_continuous_external)

    embed_t_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_local)(t_vacation)
    embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
    embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)
    embed_t_weather = Embedding(output_dim=2, input_dim=17, input_length=len_local)(t_weather)

    t_external = merge.Concatenate(axis=-1)(
        [embed_t_holiday, embed_t_hour, embed_t_dayOfWeek, embed_t_weather, t_continuous_external])

    t_out = Flatten()(t_external)
    t_out = Dense(units=50)(t_out)
    t_out = Dropout(rate=0.1)(t_out)
    t_out = Activation('relu')(t_out)
    t_out = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(t_out)
    t_out = Activation('relu')(t_out)
    feature_external_t = Reshape((2, neighbor_slide_len, neighbor_slide_len))(t_out)

    # ==============================
    # deal with the current local flow data
    current_local_flow = \
        Input(shape=(2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='current_local_flow')
    main_input.append(current_local_flow)

    cnn_out = cnn_unit(is_BN, nb_cnn)(current_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_current_local_flow)

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # todo consider the model here
    cnn_out = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_stacked_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_stacked_local_flow)

    # ==============================
    # fuse local stacked flow feature and current local flow feature, then fuse local external feature
    local_output = []
    for feature in local_feature_list:
        local_output.append(ilayer()(feature))
    feature_local_flow = merge.Add()(local_output)

    feature_local = merge.Add()([feature_external_t, feature_local_flow])

    main_output = Activation('tanh')(feature_local)

    model = Model(inputs=main_input, outputs=main_output, name='GLST-Net_noglobal')
    return model

def glst_net_lstm(args):
    len_global, len_local, neighbor_size, is_BN, nb_res_unit, nb_cnn, dataset, width, height = \
        args['len_global'], args['len_local'], args['neighbor_size'], args['is_BN'], \
        args['nb_res_unit'], args['nb_cnn'], args['dataset'], args['width'], args['height']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

    # ==============================
    # deal with the global local external data
    if dataset == 'bj_taxi':
        g_vacation = Input(shape=(len_global,), dtype='int32', name='g_vacation')
        g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
        g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
        g_weather = Input(shape=(len_global,), dtype='int32', name='g_weather')
        g_continuous_external = Input(shape=(len_global, 2), dtype='float32', name='g_continuous_external')
        main_input.append(g_vacation)
        main_input.append(g_hour)
        main_input.append(g_dayOfWeek)
        main_input.append(g_weather)
        main_input.append(g_continuous_external)

        embed_g_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_global)(g_vacation)
        embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
        embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)
        embed_g_weather = Embedding(output_dim=2, input_dim=17, input_length=len_global)(g_weather)

        g_external = merge.Concatenate(axis=-1)(
            [embed_g_holiday, embed_g_hour, embed_g_dayOfWeek, embed_g_weather, g_continuous_external])
    else:
        g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
        g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
        main_input.append(g_hour)
        main_input.append(g_dayOfWeek)

        embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
        embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)

        g_external = merge.Concatenate(axis=-1)([embed_g_hour, embed_g_dayOfWeek])

    g_out = Flatten()(g_external)
    g_out = Dense(units=50)(g_out)
    g_out = Dropout(rate=0.1)(g_out)
    g_out = Activation('relu')(g_out)
    g_out = Dense(units=2 * width * height)(g_out)
    g_out = Activation('relu')(g_out)
    feature_external_g = Reshape((2, width, height))(g_out)

    # ==============================
    # deal with the local external data
    if dataset == 'bj_taxi':
        t_vacation = Input(shape=(len_local,), dtype='int32', name='t_vacation')
        t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
        t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
        t_weather = Input(shape=(len_local,), dtype='int32', name='t_weather')
        t_continuous_external = Input(shape=(len_local, 2), dtype='float32', name='t_continuous_external')
        main_input.append(t_vacation)
        main_input.append(t_hour)
        main_input.append(t_dayOfWeek)
        main_input.append(t_weather)
        main_input.append(t_continuous_external)

        embed_t_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_local)(t_vacation)
        embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
        embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)
        embed_t_weather = Embedding(output_dim=2, input_dim=17, input_length=len_local)(t_weather)

        t_external = merge.Concatenate(axis=-1)(
            [embed_t_holiday, embed_t_hour, embed_t_dayOfWeek, embed_t_weather, t_continuous_external])
    else:
        t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
        t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
        main_input.append(t_hour)
        main_input.append(t_dayOfWeek)

        embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
        embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)

        t_external = merge.Concatenate(axis=-1)([embed_t_hour, embed_t_dayOfWeek])

    t_out = Flatten()(t_external)
    t_out = Dense(units=50)(t_out)
    t_out = Dropout(rate=0.1)(t_out)
    t_out = Activation('relu')(t_out)
    t_out = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(t_out)
    t_out = Activation('relu')(t_out)
    feature_external_t = Reshape((2, neighbor_slide_len, neighbor_slide_len))(t_out)

    # ==============================
    # deal with the current local flow data
    current_local_flow = \
        Input(shape=(2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='current_local_flow')
    main_input.append(current_local_flow)

    cnn_out = cnn_unit(is_BN, nb_cnn)(current_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_current_local_flow)

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # cnn unit + Dense + Reshape + LSTM + Reshape
    stack_local_flow = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    stack_local_flow = Conv2D(len_local, (3, 3), strides=(1, 1), padding='same')(stack_local_flow)

    stack_local_flow = Flatten()(stack_local_flow)
    stack_local_flow = Reshape(target_shape=(len_local, -1))(stack_local_flow)
    stack_local_dense = Dense(units=64, activation='relu')(stack_local_flow)

    stack_local_lstm = LSTM(units=2 * neighbor_slide_len * neighbor_slide_len,
                            return_sequences=False, dropout=0)(stack_local_dense)

    feature_stacked_local_flow = Reshape(target_shape=(2, neighbor_slide_len, neighbor_slide_len))(stack_local_lstm)

    local_feature_list.append(feature_stacked_local_flow)

    # ==============================
    # fuse local stacked flow feature and current local flow feature, then fuse local external feature
    local_output = []
    for feature in local_feature_list:
        local_output.append(ilayer()(feature))
    feature_local_flow = merge.Add()(local_output)

    # todo consider the fuse method
    feature_local = merge.Add()([feature_external_t, feature_local_flow])

    # ==============================
    # deal with the global flow data, then fuse global external feature
    # Res unit + Dense + Reshape + LSTM + Reshape
    global_flow = \
        Input(shape=(len_global * 2, width, height), dtype='float32', name='global_flow')
    main_input.append(global_flow)

    conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(global_flow)
    activation = Activation('relu')(conv1)
    res_out = rest_unit(is_BN, nb_res_unit)(activation)
    activation = Activation('relu')(res_out)
    res_out = Conv2D(len_global, (3, 3), strides=(1, 1), padding='same')(activation)

    res_out = Flatten()(res_out)
    res_out = Reshape(target_shape=(len_global, -1))(res_out)
    dense_out = Dense(units=64, activation='relu')(res_out)

    lstm_out = LSTM(units=2 * width * height, return_sequences=False, dropout=0)(dense_out)

    feature_global_flow = Reshape(target_shape=(2, width, height))(lstm_out)

    # todo consider the fuse method
    feature_global = merge.Add()([feature_external_g, feature_global_flow])

    # reshape the global feaure
    feature_global = Flatten()(feature_global)
    feature_global = Dense(units=200)(feature_global)
    feature_global = Dropout(rate=0.1)(feature_global)
    feature_global = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(feature_global)
    feature_global = Activation('relu')(feature_global)
    feature_global = Reshape((2, neighbor_slide_len, neighbor_slide_len))(feature_global)

    # ==============================
    # fuse local and global feature to predict
    # index_cut = Input(shape=(2,), dtype='int32', name='index_cut')
    # main_input.append(index_cut)
    # row, column = index_cut[0].eval(), index_cut[1].eval(session=K.get_session())

    # fuse the global and local feature
    output = []
    for feature in [feature_global, feature_local]:
        output.append(ilayer()(feature))
    main_output = merge.Add()(output)

    main_output = Activation('tanh')(main_output)

    model = Model(inputs=main_input, outputs=main_output, name='GLST-Net')
    return model

def glst_net_convlstm(args):
    len_global, len_local, neighbor_size, is_BN, nb_res_unit, nb_cnn = \
        args['len_global'], args['len_local'], args['neighbor_size'], args['is_BN'], args['nb_res_unit'], \
        args['nb_cnn']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

    # ==============================
    # deal with the global local external data

    g_vacation = Input(shape=(len_global,), dtype='int32', name='g_vacation')
    g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
    g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
    g_weather = Input(shape=(len_global,), dtype='int32', name='g_weather')
    g_continuous_external = Input(shape=(len_global, 2), dtype='float32', name='g_continuous_external')
    main_input.append(g_vacation)
    main_input.append(g_hour)
    main_input.append(g_dayOfWeek)
    main_input.append(g_weather)
    main_input.append(g_continuous_external)

    embed_g_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_global)(g_vacation)
    embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
    embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)
    embed_g_weather = Embedding(output_dim=2, input_dim=17, input_length=len_global)(g_weather)

    g_external = merge.Concatenate(axis=-1)(
        [embed_g_holiday, embed_g_hour, embed_g_dayOfWeek, embed_g_weather, g_continuous_external])

    g_out = Flatten()(g_external)
    g_out = Dense(units=50)(g_out)
    g_out = Dropout(rate=0.1)(g_out)
    g_out = Activation('relu')(g_out)
    g_out = Dense(units=2 * 32 * 32)(g_out)
    g_out = Activation('relu')(g_out)
    feature_external_g = Reshape((2, 32, 32))(g_out)

    # ==============================
    # deal with the local external data
    t_vacation = Input(shape=(len_local,), dtype='int32', name='t_vacation')
    t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
    t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
    t_weather = Input(shape=(len_local,), dtype='int32', name='t_weather')
    t_continuous_external = Input(shape=(len_local, 2), dtype='float32', name='t_continuous_external')
    main_input.append(t_vacation)
    main_input.append(t_hour)
    main_input.append(t_dayOfWeek)
    main_input.append(t_weather)
    main_input.append(t_continuous_external)

    embed_t_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_local)(t_vacation)
    embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
    embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)
    embed_t_weather = Embedding(output_dim=2, input_dim=17, input_length=len_local)(t_weather)

    t_external = merge.Concatenate(axis=-1)(
        [embed_t_holiday, embed_t_hour, embed_t_dayOfWeek, embed_t_weather, t_continuous_external])

    t_out = Flatten()(t_external)
    t_out = Dense(units=50)(t_out)
    t_out = Dropout(rate=0.1)(t_out)
    t_out = Activation('relu')(t_out)
    t_out = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(t_out)
    t_out = Activation('relu')(t_out)
    feature_external_t = Reshape((2, neighbor_slide_len, neighbor_slide_len))(t_out)

    # ==============================
    # deal with the current local flow data
    current_local_flow = \
        Input(shape=(2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='current_local_flow')
    main_input.append(current_local_flow)

    cnn_out = cnn_unit(is_BN, nb_cnn)(current_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_current_local_flow)

    # ==============================
    # deal with the local stacked flow data
    # cnn unit + Conv2D + Reshape + ConvLSTM
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # use res unit to extract the feature first, then use convlstm
    stack_local_flow = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    stack_local_flow = Activation('relu')(stack_local_flow)
    stack_local_flow = Conv2D(2 * len_local, (3, 3), strides=(1, 1), padding='same')(stack_local_flow)

    stack_local_flow = Reshape(target_shape=(len_local, 2, neighbor_slide_len, neighbor_slide_len))(stack_local_flow)
    stack_local_flow = ConvLSTM2D(2, (3, 3), strides=(1, 1), padding='same', activation='relu',
                       return_sequences=False)(stack_local_flow)
    local_feature_list.append(stack_local_flow)

    # ==============================
    # fuse local stacked flow feature and current local flow feature, then fuse local external feature
    local_output = []
    for feature in local_feature_list:
        local_output.append(ilayer()(feature))
    feature_local_flow = merge.Add()(local_output)

    # todo consider the fuse method
    feature_local = merge.Add()([feature_external_t, feature_local_flow])

    # ==============================
    # deal with the global flow data, then fuse global external feature
    # res unit + Conv2D + Reshape + ConvLSTM
    global_flow = \
        Input(shape=(len_global * 2, 32, 32), dtype='float32', name='global_flow')
    main_input.append(global_flow)

    # use res unit to extract the feature first, then use convlstm
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(global_flow)
    activation = Activation('relu')(conv1)
    res_out = rest_unit(is_BN, nb_res_unit)(activation)
    activation = Activation('relu')(res_out)
    global_flow = Conv2D(2 * len_global, (3, 3), strides=(1, 1), padding='same')(activation)

    global_flow = Reshape(target_shape=(len_global, 2, 32, 32))(global_flow)
    feature_global_flow = ConvLSTM2D(2, (3, 3), strides=(1, 1), padding='same', activation='relu',
                                  return_sequences=False)(global_flow)

    feature_global = merge.Add()([feature_external_g, feature_global_flow])

    # reshape the global feaure
    feature_global = Flatten()(feature_global)
    feature_global = Dense(units=200)(feature_global)
    feature_global = Dropout(rate=0.1)(feature_global)
    feature_global = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(feature_global)
    feature_global = Activation('relu')(feature_global)
    feature_global = Reshape((2, neighbor_slide_len, neighbor_slide_len))(feature_global)

    # ==============================
    # fuse local and global feature to predict
    # index_cut = Input(shape=(2,), dtype='int32', name='index_cut')
    # main_input.append(index_cut)
    # row, column = index_cut[0].eval(), index_cut[1].eval(session=K.get_session())

    # fuse the global and local feature
    output = []
    for feature in [feature_global, feature_local]:
        output.append(ilayer()(feature))
    main_output = merge.Add()(output)

    main_output = Activation('tanh')(main_output)

    model = Model(inputs=main_input, outputs=main_output, name='GLST-Net')
    return model

def glst_net_lstm_double_attention(args):
    len_global, len_local, neighbor_size, is_BN, nb_res_unit, nb_cnn = \
        args['len_global'], args['len_local'], args['neighbor_size'], args['is_BN'], \
        args['nb_res_unit'], args['nb_cnn']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

    # ==============================
    # deal with the global local external data

    g_vacation = Input(shape=(len_global,), dtype='int32', name='g_vacation')
    g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
    g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
    g_weather = Input(shape=(len_global,), dtype='int32', name='g_weather')
    g_continuous_external = Input(shape=(len_global, 2), dtype='float32', name='g_continuous_external')
    main_input.append(g_vacation)
    main_input.append(g_hour)
    main_input.append(g_dayOfWeek)
    main_input.append(g_weather)
    main_input.append(g_continuous_external)

    embed_g_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_global)(g_vacation)
    embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
    embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)
    embed_g_weather = Embedding(output_dim=2, input_dim=17, input_length=len_global)(g_weather)

    g_external = merge.Concatenate(axis=-1)(
        [embed_g_holiday, embed_g_hour, embed_g_dayOfWeek, embed_g_weather, g_continuous_external])

    g_out = Flatten()(g_external)
    g_out = Dense(units=50)(g_out)
    g_out = Dropout(rate=0.1)(g_out)
    g_out = Activation('relu')(g_out)
    g_out = Dense(units=len_global * external_len)(g_out)
    g_out = Activation('relu')(g_out)
    feature_external_g = Reshape((len_global, -1))(g_out)

    # ==============================
    # deal with the local external data
    t_vacation = Input(shape=(len_local,), dtype='int32', name='t_vacation')
    t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
    t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
    t_weather = Input(shape=(len_local,), dtype='int32', name='t_weather')
    t_continuous_external = Input(shape=(len_local, 2), dtype='float32', name='t_continuous_external')
    main_input.append(t_vacation)
    main_input.append(t_hour)
    main_input.append(t_dayOfWeek)
    main_input.append(t_weather)
    main_input.append(t_continuous_external)

    embed_t_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_local)(t_vacation)
    embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
    embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)
    embed_t_weather = Embedding(output_dim=2, input_dim=17, input_length=len_local)(t_weather)

    t_external = merge.Concatenate(axis=-1)(
        [embed_t_holiday, embed_t_hour, embed_t_dayOfWeek, embed_t_weather, t_continuous_external])

    t_out = Flatten()(t_external)
    t_out = Dense(units=50)(t_out)
    t_out = Dropout(rate=0.1)(t_out)
    t_out = Activation('relu')(t_out)
    t_out = Dense(units=len_local * external_len)(t_out)
    t_out = Activation('relu')(t_out)
    feature_external_t = Reshape((len_local, -1))(t_out)

    # ==============================
    # deal with the global flow data, then compute the global attention
    # Res unit + Dense + Reshape + LSTM ==> Global Attention
    global_flow = \
        Input(shape=(len_global * 2, 32, 32), dtype='float32', name='global_flow')
    main_input.append(global_flow)

    conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(global_flow)
    activation = Activation('relu')(conv1)
    res_out = rest_unit(is_BN, nb_res_unit)(activation)
    activation = Activation('relu')(res_out)
    res_out = Conv2D(len_global, (3, 3), strides=(1, 1), padding='same')(activation)

    res_out = Flatten()(res_out)
    res_out = Reshape(target_shape=(len_global, -1))(res_out)
    global_flow_dense = Dense(units=64, activation='relu')(res_out)

    global_data = merge.Concatenate(axis=-1)([global_flow_dense, feature_external_g])

    # Global Attention here Hg
    # compute global attention
    lstm_global_seq = LSTM(units=2 * neighbor_slide_len * neighbor_slide_len,
                           return_sequences=True, dropout=0.1,
                           recurrent_dropout=0.1, name="att_global")(global_data)

    attention_hidden_global = Lambda(lambda x: x[:, :-1])(lstm_global_seq)
    target_hidden_global = Lambda(lambda x: x[:, -1])(lstm_global_seq)

    att_global = Attention()([attention_hidden_global, target_hidden_global])

    lstm_global = merge.Concatenate(axis=-1)([att_global, target_hidden_global])

    lstm_global = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(lstm_global)

    # ==============================
    # deal with the local stacked flow data
    # Cnn unit + Dense + Reshape + LSTM ==> Local Attention
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    stack_local_flow = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    stack_local_flow = Conv2D(len_local, (3, 3), strides=(1, 1), padding='same')(stack_local_flow)

    stack_local_flow = Flatten()(stack_local_flow)
    stack_local_flow = Reshape(target_shape=(len_local, -1))(stack_local_flow)
    local_flow_dense = Dense(units=64, activation='relu')(stack_local_flow)

    local_data = merge.Concatenate(axis=-1)([local_flow_dense, feature_external_t])

    # Local Attention here --> Hc
    lstm_local_seq = LSTM(units=2 * neighbor_slide_len * neighbor_slide_len,
                            return_sequences=True, dropout=0.1,
                            recurrent_dropout=0.1, name="att_local")(local_data)

    attention_hidden_local = Lambda(lambda x: x[:, :-1])(lstm_local_seq)
    target_hidden_local = Lambda(lambda x: x[:, -1])(lstm_local_seq)

    att_local = Attention()([attention_hidden_local, target_hidden_local])

    # ==============================
    # fuse the global and local attention, then combine target hidden state to predict
    output = []
    for feature in [lstm_global, att_local]:
        output.append(ilayer()(feature))
    att_final = merge.Add()(output)

    lstm_all = merge.Concatenate(axis=-1)([att_final, target_hidden_local])
    lstm_all = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(lstm_all)
    lstm_all = Reshape((2, neighbor_slide_len, neighbor_slide_len))(lstm_all)

    main_output = Activation('tanh')(lstm_all)

    model = Model(inputs=main_input, outputs=main_output, name='GLST-Net-LSTM-DA')
    return model

def glst_net_lstm_local_attention(args):
    len_global, len_local, neighbor_size, is_BN, nb_res_unit, nb_cnn = \
        args['len_global'], args['len_local'], args['neighbor_size'], args['is_BN'], \
        args['nb_res_unit'], args['nb_cnn']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    # ==============================
    # deal with the global local external data

    g_vacation = Input(shape=(len_global,), dtype='int32', name='g_vacation')
    g_hour = Input(shape=(len_global,), dtype='int32', name='g_hour')
    g_dayOfWeek = Input(shape=(len_global,), dtype='int32', name='g_dayOfWeek')
    g_weather = Input(shape=(len_global,), dtype='int32', name='g_weather')
    g_continuous_external = Input(shape=(len_global, 2), dtype='float32', name='g_continuous_external')
    main_input.append(g_vacation)
    main_input.append(g_hour)
    main_input.append(g_dayOfWeek)
    main_input.append(g_weather)
    main_input.append(g_continuous_external)

    embed_g_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_global)(g_vacation)
    embed_g_hour = Embedding(output_dim=2, input_dim=25, input_length=len_global)(g_hour)
    embed_g_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_global)(g_dayOfWeek)
    embed_g_weather = Embedding(output_dim=2, input_dim=17, input_length=len_global)(g_weather)

    g_external = merge.Concatenate(axis=-1)(
        [embed_g_holiday, embed_g_hour, embed_g_dayOfWeek, embed_g_weather, g_continuous_external])

    g_out = Flatten()(g_external)
    g_out = Dense(units=50)(g_out)
    g_out = Dropout(rate=0.1)(g_out)
    g_out = Activation('relu')(g_out)
    g_out = Dense(units=2 * 32 * 32)(g_out)
    g_out = Activation('relu')(g_out)
    feature_external_g = Reshape((2, 32, 32))(g_out)

    # ==============================
    # deal with the local external data
    t_vacation = Input(shape=(len_local,), dtype='int32', name='t_vacation')
    t_hour = Input(shape=(len_local,), dtype='int32', name='t_hour')
    t_dayOfWeek = Input(shape=(len_local,), dtype='int32', name='t_dayOfWeek')
    t_weather = Input(shape=(len_local,), dtype='int32', name='t_weather')
    t_continuous_external = Input(shape=(len_local, 2), dtype='float32', name='t_continuous_external')
    main_input.append(t_vacation)
    main_input.append(t_hour)
    main_input.append(t_dayOfWeek)
    main_input.append(t_weather)
    main_input.append(t_continuous_external)

    embed_t_holiday = Embedding(output_dim=2, input_dim=2, input_length=len_local)(t_vacation)
    embed_t_hour = Embedding(output_dim=2, input_dim=25, input_length=len_local)(t_hour)
    embed_t_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_local)(t_dayOfWeek)
    embed_t_weather = Embedding(output_dim=2, input_dim=17, input_length=len_local)(t_weather)

    t_external = merge.Concatenate(axis=-1)(
        [embed_t_holiday, embed_t_hour, embed_t_dayOfWeek, embed_t_weather, t_continuous_external])

    t_out = Flatten()(t_external)
    t_out = Dense(units=50)(t_out)
    t_out = Dropout(rate=0.1)(t_out)
    t_out = Activation('relu')(t_out)
    t_out = Dense(units=len_local * external_len)(t_out)
    t_out = Activation('relu')(t_out)
    feature_external_t = Reshape((len_local, -1))(t_out)

    # ==============================
    # deal with the global flow data, then fuse global external feature
    # Res unit + Dense + Reshape + LSTM + Reshape
    global_flow = \
        Input(shape=(len_global * 2, 32, 32), dtype='float32', name='global_flow')
    main_input.append(global_flow)

    conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(global_flow)
    activation = Activation('relu')(conv1)
    res_out = rest_unit(is_BN, nb_res_unit)(activation)
    activation = Activation('relu')(res_out)
    res_out = Conv2D(len_global, (3, 3), strides=(1, 1), padding='same')(activation)

    res_out = Flatten()(res_out)
    res_out = Reshape(target_shape=(len_global, -1))(res_out)
    dense_out = Dense(units=64, activation='relu')(res_out)

    lstm_out = LSTM(units=2 * 32 * 32, return_sequences=False, dropout=0)(dense_out)
    feature_global_flow = Reshape(target_shape=(2, 32, 32))(lstm_out)

    feature_global = merge.Add()([feature_external_g, feature_global_flow])

    # reshape the global feaure
    feature_global = Flatten()(feature_global)
    feature_global = Dense(units=200)(feature_global)
    feature_global = Dropout(rate=0.1)(feature_global)
    feature_global = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(feature_global)
    feature_global = Activation('relu')(feature_global)
    feature_global = Reshape((2, neighbor_slide_len, neighbor_slide_len))(feature_global)

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # cnn unit + Dense + Reshape + LSTM + Reshape
    stack_local_flow = cnn_unit(is_BN, nb_cnn)(stack_local_flow)
    stack_local_flow = Conv2D(len_local, (3, 3), strides=(1, 1), padding='same')(stack_local_flow)

    stack_local_flow = Flatten()(stack_local_flow)
    stack_local_flow = Reshape(target_shape=(len_local, -1))(stack_local_flow)
    local_flow_dense = Dense(units=64, activation='relu')(stack_local_flow)

    local_data = merge.Concatenate(axis=-1)([local_flow_dense, feature_external_t])

    # Local Attention here --> Hc
    lstm_local_seq = LSTM(units=2 * neighbor_slide_len * neighbor_slide_len,
                          return_sequences=True, dropout=0.1,
                          recurrent_dropout=0.1, name="att_local")(local_data)
    attention_hidden_local = Lambda(lambda x: x[:, :-1])(lstm_local_seq)
    target_hidden_local = Lambda(lambda x: x[:, -1])(lstm_local_seq)
    att_local = Attention()([attention_hidden_local, target_hidden_local])
    lstm_local = merge.Concatenate(axis=-1)([att_local, target_hidden_local])

    # reshape the local feature
    lstm_local = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(lstm_local)
    feature_local = Reshape((2, neighbor_slide_len, neighbor_slide_len))(lstm_local)

    # ==============================
    # fuse local and global feature to predict
    output = []
    for feature in [feature_global, feature_local]:
        output.append(ilayer()(feature))
    main_output = merge.Add()(output)

    main_output = Activation('tanh')(main_output)

    model = Model(inputs=main_input, outputs=main_output, name='GLST-Net')
    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = {'len_global': 7, 'len_local': 4, 'neighbor_size': 2, 'is_BN': True, 'nb_res_unit': 12, 'nb_cnn': 6}
    model = glst_net_lstm_local_attention(args)
    model.summary()