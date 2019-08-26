# -*- coding: utf-8 -*-
"""
Baseline HA: Average the stack and current
"""
from keras.layers import (Input, Lambda, merge)
from keras import backend as K
from keras.models import Model
import tensorflow as tf

def ha(args):
    len_local, neighbor_size = args['len_local'], args['neighbor_size']

    main_input = []
    neighbor_slide_len = neighbor_size * 2 + 1

    stack_local_flow = \
        Input(shape=(len_local * 2, neighbor_slide_len, neighbor_slide_len), dtype='float32', name='stack_local_flow')
    main_input.append(stack_local_flow)

    # merge layer ask the number greater than 1
    if len_local == 1:
        output = main_input
        model = Model(inputs=main_input, outputs=output, name='HA')
        return model

    # average the stack

    def slice(inputs, channel):
        return inputs[:, channel:channel+1, :, :]

    outflow_list = []
    inflow_list = []

    for i in range(len_local):
        inflow_list.append(Lambda(slice, arguments={'channel':2*i})(stack_local_flow))
        outflow_list.append(Lambda(slice, arguments={'channel': 2*i+1})(stack_local_flow))

    inflow_ave = merge.Average()(inflow_list)
    outflow_ave = merge.Average()(outflow_list)

    output = merge.Concatenate(axis=1)([inflow_ave, outflow_ave])

    model = Model(inputs=main_input, outputs=output, name='HA')
    return model

if __name__ == '__main__':
    model = ha({'len_local': 4, 'neighbor_size': 2})
    model.summary()