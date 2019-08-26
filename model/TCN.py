# -*- coding=utf-8 -*-

from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import (
Input, Activation, merge, Dense, Reshape, Embedding, Flatten, Dropout,
Lambda, LSTM, ConvLSTM2D, Concatenate, Conv1D, SpatialDropout1D, add
)
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import os
from model.GLSTModel import cnn_unit, ilayer, rest_unit, Attention

external_len = 10


def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0):
    # type: (Layer, int, int, int, str, float) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN
    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding=padding)(x)
        # x = BatchNormalization()(x)  # TODO should be WeightNorm here.
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = add([prev_x, x])
    return res_x, x

def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        # print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations

class TCN:
    """Creates a TCN layer.
        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).
        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        # 1D FCN.
        x = Conv1D(self.nb_filters, 1, padding=self.padding)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                x, skip_out = residual_block(x,
                                             dilation_rate=d,
                                             nb_filters=self.nb_filters,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             dropout_rate=self.dropout_rate)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = add(skip_connections)
        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        return x

def simple_tcn(args):
    len_local, neighbor_size, num_tcn, nb_stacks, dataset = \
        args['len_local'], args['neighbor_size'], args['num_tcn'], args['nb_stacks'], args['dataset']
    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

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
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)
    tcn_out = Reshape((len_local, 2 * neighbor_slide_len * neighbor_slide_len))(stack_local_flow)

    for i in range(num_tcn):
        if i == num_tcn - 1 or (i == 0 and num_tcn == 1):
            tcn_out = TCN(nb_stacks=nb_stacks, return_sequences=False)(tcn_out)
        else:
            tcn_out = TCN(nb_stacks=nb_stacks, return_sequences=True)(tcn_out)

    o = Dense(units=(2 * neighbor_slide_len * neighbor_slide_len))(tcn_out)
    feature_stacked_local_flow = Reshape((2, neighbor_slide_len, neighbor_slide_len))(o)

    feature_local = merge.Add()([feature_external_t, feature_stacked_local_flow])

    main_output = Activation('tanh')(feature_local)

    model = Model(inputs=main_input, outputs=main_output, name='simple-tcn')
    return model

def tcn_no_global(args):
    len_local, neighbor_size, num_tcn, nb_stacks, is_BN, num_cnn, dataset  = \
        args['len_local'], args['neighbor_size'], args['num_tcn'], args['nb_stacks'], \
        args['is_BN'], args['num_cnn'], args['dataset']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

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

    cnn_out = cnn_unit(is_BN, num_cnn)(current_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_current_local_flow)

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)
    tcn_out = Reshape((len_local, 2 * neighbor_slide_len * neighbor_slide_len))(stack_local_flow)

    for i in range(num_tcn):
        if i == num_tcn - 1 or (i == 0 and num_tcn == 1):
            tcn_out = TCN(nb_stacks=nb_stacks, return_sequences=False)(tcn_out)
        else:
            tcn_out = TCN(nb_stacks=nb_stacks, return_sequences=True)(tcn_out)

    o = Dense(units=(2 * neighbor_slide_len * neighbor_slide_len))(tcn_out)
    feature_stacked_local_flow = Reshape((2, neighbor_slide_len, neighbor_slide_len))(o)

    local_feature_list.append(feature_stacked_local_flow)

    # ==============================
    # fuse local stacked flow feature and current local flow feature, then fuse local external feature
    local_output = []
    for feature in local_feature_list:
        local_output.append(ilayer()(feature))
    feature_local_flow = merge.Add()(local_output)

    feature_local = merge.Add()([feature_external_t, feature_local_flow])

    main_output = Activation('tanh')(feature_local)

    model = Model(inputs=main_input, outputs=main_output, name='tcn_nog')
    return model

def tcn_nog_rdw_att(args):
    len_recent, len_daily, len_week, neighbor_size, num_tcn, nb_stacks, is_BN, num_cnn, dataset  = \
        args['len_recent'], args['len_daily'], args['len_week'], args['neighbor_size'], \
        args['num_tcn'], args['nb_stacks'], args['BN'], args['num_cnn'], args['dataset']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    local_feature_list = []

    # ==============================
    # deal with the local external data
    if dataset == 'bj_taxi':
        # current
        current_vacation = Input(shape=(1,), dtype='int32', name='current_vacation')
        current_hour = Input(shape=(1,), dtype='int32', name='current_hour')
        current_dayOfWeek = Input(shape=(1,), dtype='int32', name='current_dayOfWeek')
        current_weather = Input(shape=(1,), dtype='int32', name='current_weather')
        current_continuous_external = Input(shape=(1, 2), dtype='float32', name='current_continuous_external')

        main_input.append(current_vacation)
        main_input.append(current_hour)
        main_input.append(current_dayOfWeek)
        main_input.append(current_weather)
        main_input.append(current_continuous_external)

        embed_current_vacation = Embedding(output_dim=2, input_dim=2, input_length=1)(current_vacation)
        embed_current_hour = Embedding(output_dim=2, input_dim=25, input_length=1)(current_hour)
        embed_current_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=1)(current_dayOfWeek)
        embed_current_weather = Embedding(output_dim=2, input_dim=17, input_length=1)(current_weather)
        current_external = merge.Concatenate(axis=-1)(
            [embed_current_vacation, embed_current_hour, embed_current_dayOfWeek, embed_current_weather,
             current_continuous_external])

        # recent external
        recent_vacation = Input(shape=(len_recent,), dtype='int32', name='recent_vacation')
        recent_hour = Input(shape=(len_recent,), dtype='int32', name='recent_hour')
        recent_dayOfWeek = Input(shape=(len_recent,), dtype='int32', name='recent_dayOfWeek')
        recent_weather = Input(shape=(len_recent,), dtype='int32', name='recent_weather')
        recent_continuous_external = Input(shape=(len_recent, 2), dtype='float32', name='recent_continuous_external')

        main_input.append(recent_vacation)
        main_input.append(recent_hour)
        main_input.append(recent_dayOfWeek)
        main_input.append(recent_weather)
        main_input.append(recent_continuous_external)

        embed_recent_vacation = Embedding(output_dim=2, input_dim=2, input_length=len_recent)(recent_vacation)
        embed_recent_hour = Embedding(output_dim=2, input_dim=25, input_length=len_recent)(recent_hour)
        embed_recent_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_recent)(recent_dayOfWeek)
        embed_recent_weather = Embedding(output_dim=2, input_dim=17, input_length=len_recent)(recent_weather)
        recent_external = merge.Concatenate(axis=-1)(
            [embed_recent_vacation, embed_recent_hour, embed_recent_dayOfWeek, embed_recent_weather, recent_continuous_external])

        # daily external
        daily_vacation = Input(shape=(len_daily,), dtype='int32', name='daily_vacation')
        daily_hour = Input(shape=(len_daily,), dtype='int32', name='daily_hour')
        daily_dayOfWeek = Input(shape=(len_daily,), dtype='int32', name='daily_dayOfWeek')
        daily_weather = Input(shape=(len_daily,), dtype='int32', name='daily_weather')
        daily_continuous_external = Input(shape=(len_daily, 2), dtype='float32', name='daily_continuous_external')

        main_input.append(daily_vacation)
        main_input.append(daily_hour)
        main_input.append(daily_dayOfWeek)
        main_input.append(daily_weather)
        main_input.append(daily_continuous_external)

        embed_daily_vacation = Embedding(output_dim=2, input_dim=2, input_length=len_daily)(daily_vacation)
        embed_daily_hour = Embedding(output_dim=2, input_dim=25, input_length=len_daily)(daily_hour)
        embed_daily_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_daily)(daily_dayOfWeek)
        embed_daily_weather = Embedding(output_dim=2, input_dim=17, input_length=len_daily)(daily_weather)
        daily_external = merge.Concatenate(axis=-1)(
            [embed_daily_vacation, embed_daily_hour, embed_daily_dayOfWeek, embed_daily_weather, daily_continuous_external])

        # weekly external
        week_vacation = Input(shape=(len_week,), dtype='int32', name='week_vacation')
        week_hour = Input(shape=(len_week,), dtype='int32', name='week_hour')
        week_dayOfWeek = Input(shape=(len_week,), dtype='int32', name='week_dayOfWeek')
        week_weather = Input(shape=(len_week,), dtype='int32', name='week_weather')
        week_continuous_external = Input(shape=(len_week, 2), dtype='float32', name='week_continuous_external')

        main_input.append(week_vacation)
        main_input.append(week_hour)
        main_input.append(week_dayOfWeek)
        main_input.append(week_weather)
        main_input.append(week_continuous_external)

        embed_week_vacation = Embedding(output_dim=2, input_dim=2, input_length=len_week)(week_vacation)
        embed_week_hour = Embedding(output_dim=2, input_dim=25, input_length=len_week)(week_hour)
        embed_week_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_week)(week_dayOfWeek)
        embed_week_weather = Embedding(output_dim=2, input_dim=17, input_length=len_week)(week_weather)
        week_external = merge.Concatenate(axis=-1)(
            [embed_week_vacation, embed_week_hour, embed_week_dayOfWeek, embed_week_weather,
             week_continuous_external])
    else:
        # current
        current_hour = Input(shape=(1,), dtype='int32', name='current_hour')
        current_dayOfWeek = Input(shape=(1,), dtype='int32', name='current_dayOfWeek')

        main_input.append(current_hour)
        main_input.append(current_dayOfWeek)

        embed_current_hour = Embedding(output_dim=2, input_dim=25, input_length=1)(current_hour)
        embed_current_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=1)(current_dayOfWeek)
        current_external = merge.Concatenate(axis=-1)([embed_current_hour, embed_current_dayOfWeek])

        # recent external
        recent_hour = Input(shape=(len_recent,), dtype='int32', name='recent_hour')
        recent_dayOfWeek = Input(shape=(len_recent,), dtype='int32', name='recent_dayOfWeek')

        main_input.append(recent_hour)
        main_input.append(recent_dayOfWeek)

        embed_recent_hour = Embedding(output_dim=2, input_dim=25, input_length=len_recent)(recent_hour)
        embed_recent_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_recent)(recent_dayOfWeek)

        recent_external = merge.Concatenate(axis=-1)([embed_recent_hour, embed_recent_dayOfWeek])

        # daily external
        daily_hour = Input(shape=(len_daily,), dtype='int32', name='daily_hour')
        daily_dayOfWeek = Input(shape=(len_daily,), dtype='int32', name='daily_dayOfWeek')

        main_input.append(daily_hour)
        main_input.append(daily_dayOfWeek)

        embed_daily_hour = Embedding(output_dim=2, input_dim=25, input_length=len_daily)(daily_hour)
        embed_daily_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_daily)(daily_dayOfWeek)
        daily_external = merge.Concatenate(axis=-1)([embed_daily_hour, embed_daily_dayOfWeek])

        # weekly external
        week_hour = Input(shape=(len_week,), dtype='int32', name='week_hour')
        week_dayOfWeek = Input(shape=(len_week,), dtype='int32', name='week_dayOfWeek')

        main_input.append(week_hour)
        main_input.append(week_dayOfWeek)

        embed_week_hour = Embedding(output_dim=2, input_dim=25, input_length=len_week)(week_hour)
        embed_week_dayOfWeek = Embedding(output_dim=2, input_dim=7, input_length=len_week)(week_dayOfWeek)

        week_external = merge.Concatenate(axis=-1)([embed_week_hour, embed_week_dayOfWeek])

    current_out = Flatten()(current_external)
    current_out = Dense(units=20)(current_out)
    current_out = Dropout(rate=0.1)(current_out)
    current_out = Activation('relu')(current_out)
    current_out = Dense(units=10)(current_out)
    feature_external_current = Activation('relu')(current_out)

    recent_out = Dense(units=20)(recent_external)
    recent_out = Dropout(rate=0.1)(recent_out)
    recent_out = Activation('relu')(recent_out)
    recent_out = Dense(units=10)(recent_out)
    feature_external_recent = Activation('relu')(recent_out)

    daily_out = Dense(units=20)(daily_external)
    daily_out = Dropout(rate=0.1)(daily_out)
    daily_out = Activation('relu')(daily_out)
    daily_out = Dense(units=10)(daily_out)
    feature_external_daily = Activation('relu')(daily_out)

    week_out = Dense(units=20)(week_external)
    week_out = Dropout(rate=0.1)(week_out)
    week_out = Activation('relu')(week_out)
    week_out = Dense(units=10)(week_out)
    feature_external_week = Activation('relu')(week_out)

    # ==============================
    # deal with the current local flow data
    current_local_flow = \
        Input(shape=(2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='current_local_flow')
    main_input.append(current_local_flow)

    current_local_flow = cnn_unit(is_BN, num_cnn)(current_local_flow)
    current_local_flow = Activation('relu')(current_local_flow)
    current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(current_local_flow)

    current_local_flow = Flatten()(current_local_flow)
    current_local_flow = merge.Concatenate(axis=-1)([feature_external_current, current_local_flow])
    current_local_flow = Dense(units=64, activation='relu')(current_local_flow)

    local_feature_list.append(current_local_flow)

    # ==============================
    # deal with the recent stacked flow data
    recent_local_flow = \
        Input(shape=(len_recent * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='recent_local_flow')
    main_input.append(recent_local_flow)
    recent_local_flow = Reshape((len_recent, 2 * neighbor_slide_len * neighbor_slide_len))(recent_local_flow)
    recent_local_flow = merge.Concatenate(axis=-1)([feature_external_recent, recent_local_flow])
    recent_local_flow = Dense(units=64, activation='relu')(recent_local_flow)

    # deal with the daily stacked flow data
    daily_local_flow = \
        Input(shape=(len_daily * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32',
              name='daily_local_flow')
    main_input.append(daily_local_flow)
    daily_local_flow = Reshape((len_daily, 2 * neighbor_slide_len * neighbor_slide_len))(daily_local_flow)
    daily_local_flow = merge.Concatenate(axis=-1)([feature_external_daily, daily_local_flow])
    daily_local_flow = Dense(units=64, activation='relu')(daily_local_flow)

    # deal with the week stacked flow data
    week_local_flow = \
        Input(shape=(len_week * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32',
              name='week_local_flow')
    main_input.append(week_local_flow)
    week_local_flow = Reshape((len_week, 2 * neighbor_slide_len * neighbor_slide_len))(week_local_flow)
    week_local_flow = merge.Concatenate(axis=-1)([feature_external_week, week_local_flow])
    week_local_flow = Dense(units=64, activation='relu')(week_local_flow)

    for i in range(num_tcn):
        recent_local_flow = TCN(nb_stacks=nb_stacks, return_sequences=True)(recent_local_flow)
        daily_local_flow = TCN(nb_stacks=nb_stacks, return_sequences=True)(daily_local_flow)
        week_local_flow = TCN(nb_stacks=nb_stacks, return_sequences=True)(daily_local_flow)

    attention_hidden_recent = Lambda(lambda x: x[:, :])(recent_local_flow)
    attention_hidden_daily = Lambda(lambda x: x[:, :])(daily_local_flow)
    attention_hidden_week = Lambda(lambda x: x[:, :])(week_local_flow)

    att_recent = Attention()([attention_hidden_recent, current_local_flow])
    att_daily = Attention()([attention_hidden_daily, current_local_flow])
    att_week = Attention()([attention_hidden_week, current_local_flow])

    # fuse attention
    output = []
    for att in [att_recent, att_daily, att_week]:
        output.append(ilayer()(att))
    att_final = merge.Add()(output)

    att_current = merge.Concatenate(axis=-1)([att_final, current_local_flow])

    att_current = Dense(units=2 * neighbor_slide_len * neighbor_slide_len)(att_current)
    att_current = Reshape((2, neighbor_slide_len, neighbor_slide_len))(att_current)
    main_output = Activation('tanh')(att_current)

    model = Model(inputs=main_input, outputs=main_output, name='tcn_nog_rdw_att')
    return model

def tcn(args):
    len_global, len_local, neighbor_size, is_BN, nb_res_unit, num_cnn, num_tcn, nb_stacks, dataset, width, height = \
        args['len_global'], args['len_local'], args['neighbor_size'], args['is_BN'], \
        args['nb_res_unit'], args['num_cnn'], args['num_tcn'], args['nb_stacks'], \
        args['dataset'], args['width'], args['height']

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

    cnn_out = cnn_unit(is_BN, num_cnn)(current_local_flow)
    activation = Activation('relu')(cnn_out)
    feature_current_local_flow = Conv2D(2, (3, 3), strides=(1, 1), padding='same')(activation)

    local_feature_list.append(feature_current_local_flow)

    # ==============================
    # deal with the local stacked flow data
    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)
    tcn_out = Reshape((len_local, 2 * neighbor_slide_len * neighbor_slide_len))(stack_local_flow)

    for i in range(num_tcn):
        if i == num_tcn - 1 or (i == 0 and num_tcn == 1):
            tcn_out = TCN(nb_stacks=nb_stacks, return_sequences=False)(tcn_out)
        else:
            tcn_out = TCN(nb_stacks=nb_stacks, return_sequences=True)(tcn_out)

    o = Dense(units=(2 * neighbor_slide_len * neighbor_slide_len))(tcn_out)
    feature_stacked_local_flow = Reshape((2, neighbor_slide_len, neighbor_slide_len))(o)

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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    """
    args = {'len_recent': 3, 'len_daily': 3, 'len_week': 3, 'neighbor_size': 3, 'num_tcn': 2,
            'nb_stacks': 2, 'num_cnn': 6, 'is_BN': True, 'dataset': 'bj_taxi'}
    model = tcn_nog_rdw_att(args)
    model.summary()
    """

    args = {'len_global': 4, 'len_local': 4,  'neighbor_size': 3, 'num_tcn': 2, 'nb_res_unit': 12,
            'nb_stacks': 2, 'num_cnn': 6, 'is_BN': True, 'dataset': 'bj_taxi', 'width': 32, 'height': 32}
    model = tcn(args)
    model.summary()