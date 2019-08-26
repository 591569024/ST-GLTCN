# -*- coding:utf-8 -*-
import numpy as np
from scipy.sparse.linalg import eigs

from keras.engine.topology import Layer
from keras import backend as K
from keras.models import Model
from keras.layers import (
Input, Activation, merge, Dense, Reshape, Embedding, Flatten,
Dropout, Lambda, LSTM, ConvLSTM2D, Concatenate, Conv2D
)
import pandas as pd
import os

def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    distance_df = pd.read_csv(distance_df_filename, dtype={'from': 'int', 'to': 'int'})
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    # Fills cells in the matrix with distances.
    for index, row in enumerate(distance_df.values):
        i, j = int(row[0]), int(row[1])
        if row[2] == 0:
            continue
        A[i, j] = 1
    return A

def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: np.ndarray, shape is (N, N), N is the num of vertices
    :return: scaled_Laplacian, np.ndarray, shape (N, N)
    """
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    :param L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    :param K: the maximum order of chebyshev polynomials
    :return: cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def get_cheb_polynomials(file_path, num_of_vertices, K):
    adj_mx = get_adjacency_matrix(file_path, num_of_vertices)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = np.array(cheb_polynomial(L_tilde, K))
    return cheb_polynomials

class Spatial_Attention_layer(Layer):

    def __init__(self, **kwargs):
        super(Spatial_Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        5 param, S = V . activation((X W_1) W_2 (W_3 X)T + b)
        :param input_shape: batch, N, C, T (C is the number of feature)
        :return:
        """

        # get shape of input matrix x
        self.num_of_vertices = input_shape[1]
        self.num_of_features = input_shape[2]
        self.num_of_timesteps = input_shape[3]
        #self.W_1.shape = (num_of_timesteps,)
        #self.W_2.shape = (num_of_features, num_of_timesteps)
        #self.W_3.shape = (num_of_features,)
        #self.b_s.shape = (1, num_of_vertices, num_of_vertices)
        #self.V_s.shape = (num_of_vertices, num_of_vertices)

        self.w1 = self.add_weight(name='w1', shape=(self.num_of_timesteps, ),
                                  initializer='glorot_normal', trainable=True)
        self.w2 = self.add_weight(name='w2', shape=(self.num_of_features, self.num_of_timesteps),
                                  initializer='glorot_normal', trainable=True)
        self.w3 = self.add_weight(name='w3', shape=(self.num_of_features,),
                                  initializer='glorot_normal', trainable=True)
        self.b = self.add_weight(name='b', shape=(1, self.num_of_vertices, self.num_of_vertices),
                                 initializer='zeros', trainable=True)
        self.v = self.add_weight(name='v', shape=(self.num_of_vertices, self.num_of_vertices),
                                 initializer='glorot_normal', trainable=True)

        super(Spatial_Attention_layer, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        :param inputs: x^{(r - 1)}_h, shape is (batch_size, N, C_{r-1}, T_{r-1}) ?, 10, 2, 20
        :param mask:
        :return: S', spatial attention scores, shape is (batch_size, N, N)
        '''

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        x_w1 = K.squeeze(K.dot(inputs, K.expand_dims(self.w1)), axis=-1)
        lhs = K.dot(x_w1, self.w2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.squeeze(K.dot(K.permute_dimensions(inputs, (0, 1, 3, 2)), K.expand_dims(self.w3, axis=-1)), axis=-1)
        rhs = K.permute_dimensions(rhs, (0, 2, 1))

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        # shape S is (batch, V, V)
        S = K.permute_dimensions(K.dot(self.v, K.permute_dimensions(K.sigmoid(product + self.b), (1, 2, 0))), (2, 0, 1))

        # normalization
        S = S - K.max(S, axis=-1, keepdims=True)
        exp = K.exp(S)
        sum = K.sum(exp, axis=-1, keepdims=True)
        sum_list = [sum for i in range(exp.shape[-1])]
        sum_equal = Concatenate(axis=-1)(sum_list)
        S_normalized = exp / sum_equal
        #print("sat", exp.shape, sum_equal.shape, S_normalized.shape)
        return S_normalized

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        num_of_vertices = input_shape[1]
        return batch, num_of_vertices, num_of_vertices

class cheb_conv_with_SAt(Layer):

    def __init__(self, K, num_of_filters, cheb_polynomials, **kwargs):
        super(cheb_conv_with_SAt, self).__init__(**kwargs)

        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.K = K


    def build(self, input_shape):
        # shape is (batch_size, N, F, T_{r-1}), F is the num of features
        self.batch_size = input_shape[0][0]
        self.num_of_vertices = input_shape[0][1]
        self.num_of_features = input_shape[0][2]
        self.num_of_timesteps = input_shape[0][3]
        # self.Theta.shape = (self.K, num_of_features, self.num_of_filters)

        self.theta = self.add_weight(name='theta', shape=(self.K, self.num_of_features, self.num_of_filters),
                                  initializer='glorot_normal', trainable=True)
        super(cheb_conv_with_SAt, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        Chebyshev graph convolution operation
        :param inputs:
            input[0]. graph signal matrix, shape is (batch_size, N, F, T_{r-1}), F is the num of features
            input[1]. spatial_attention: shape is (batch_size, N, N), spatial attention scores
        :param mask:
        :return: convolution result, shape is (batch_size, N, self.num_of_filters, T_{r-1})
        """
        graph_signal = inputs[0]
        spatial_attention = inputs[1]
        outputs = []
        for time_step in range(graph_signal.shape[3]):
            # shape is (batch_size, V, F)
            graph_signal_tmp = graph_signal[:, :, :, time_step]
            output = K.zeros(shape=(self.num_of_vertices, self.num_of_filters))
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.theta[k]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(K.permute_dimensions(T_k_with_at, (0, 2, 1)), graph_signal_tmp)

                output = output + K.dot(rhs, theta_k)
            # 最后增加一维
            outputs.append(K.expand_dims(output, axis=-1))

        tmp = Concatenate(axis=-1)([*outputs]) if len(outputs) > 1 else outputs[0]

        return K.relu(tmp)

    def compute_output_shape(self, input_shape):
        # convolution result, shape is (batch_size, N, self.num_of_filters, T_{r-1})
        return self.batch_size, self.num_of_vertices, self.num_of_filters, self.num_of_timesteps

class Temporal_Attention_layer(Layer):

    def __init__(self, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        5 param, S = V . activation((X U_1) U_2 (U_3 X)T + b)
        :param input_shape: batch, N, C, T (C is the number of feature)
        :return:
        """

        # get shape of input matrix x
        num_of_vertices = input_shape[1]
        num_of_features = input_shape[2]
        num_of_timesteps = input_shape[3]
        #self.U_1.shape = (num_of_vertices,)
        #self.U_2.shape = (num_of_features, num_of_vertices)
        #self.U_3.shape = (num_of_features,)
        #self.b_e.shape = (1, num_of_timesteps, num_of_timesteps)
        #self.V_e.shape = (num_of_timesteps, num_of_timesteps)

        self.u1 = self.add_weight(name='u1', shape=(num_of_vertices, ),
                                  initializer='glorot_normal', trainable=True)
        self.u2 = self.add_weight(name='u2', shape=(num_of_features, num_of_vertices),
                                  initializer='glorot_normal', trainable=True)
        self.u3 = self.add_weight(name='u3', shape=(num_of_features,),
                                  initializer='glorot_normal', trainable=True)
        self.b = self.add_weight(name='b', shape=(1, num_of_timesteps, num_of_timesteps),
                                 initializer='zeros', trainable=True)
        self.v = self.add_weight(name='v', shape=(num_of_timesteps, num_of_timesteps),
                                 initializer='glorot_normal', trainable=True)

        super(Temporal_Attention_layer, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        :param inputs: x^{(r - 1)}_h, shape is (batch_size, N, C_{r-1}, T_{r-1}) ?, 10, 2, 20
        :param mask:
        :return: S', temporal attention scores, shape is (batch_size, T_{r-1}, T_{r-1})
        '''

        # compute temporal attention scores
        # shape is (N, T, V)
        print('temporal input', inputs.shape)
        x_u1 = K.squeeze(K.dot(K.permute_dimensions(inputs, (0, 3, 2, 1)), K.expand_dims(self.u1, axis=-1)), axis=-1)
        lhs = K.dot(x_u1, self.u2)

        # shape is (N, V, T)
        rhs = K.squeeze(K.dot(K.permute_dimensions(inputs, (0, 1, 3, 2)), K.expand_dims(self.u3, axis=-1)), axis=-1)
        #print('input', inputs.shape, 'lhs', lhs.shape, 'rhs', rhs.shape)

        # shape is (N, T, T)
        # todo here production size change, because time step change, but [self.v, self.b] didn't change
        product = K.batch_dot(lhs, rhs)
        #print('product', product.shape)

        E = K.permute_dimensions(K.dot(self.v, K.permute_dimensions(K.sigmoid(product + self.b), (1, 2, 0))), (2, 0, 1))

        # normailzation
        E = E - K.max(E, axis=-1, keepdims=True)
        exp = K.exp(E)
        sum = K.sum(exp, axis=-1, keepdims=True)
        # concate 1-->original dimension
        sum_list = [sum for i in range(exp.shape[-1])]
        sum_equal = Concatenate(axis=-1)(sum_list)
        E_normalized = exp / sum_equal

        #print('temporal here', exp.shape, sum_equal.shape, E_normalized.shape)
        return E_normalized

    def compute_output_shape(self, input_shape):
        # temporal attention scores, shape is (batch_size, T_{r-1}, T_{r-1})
        return input_shape[0], input_shape[-1], input_shape[-1]

class ASTGCN_block(Layer):

    def __init__(self, params, **kwargs):
        super(ASTGCN_block, self).__init__(**kwargs)
        self.K = params['K']
        self.num_of_chev_filters = params['num_of_chev_filters']
        self.num_of_time_filters = params['num_of_time_filters']
        self.time_conv_strides = params['time_conv_strides']
        self.cheb_polynomials = params["cheb_polynomials"]

        self.SAt = Spatial_Attention_layer(name="sat")
        self.cheb_conv_SAt = cheb_conv_with_SAt(num_of_filters=self.num_of_chev_filters, K=self.K,
                                                cheb_polynomials=self.cheb_polynomials, name="cheb_sat")
        self.TAt = Temporal_Attention_layer(name="{}_tat".format(self.name))
        self.time_conv = Conv2D(filters=self.num_of_time_filters, kernel_size=(1, 3), padding="same",
                                   strides=(1, 1), name="{}_time_conv2d".format(self.name))
        self.residual_conv = Conv2D(filters=self.num_of_time_filters, kernel_size=(1, 1),
                                       strides=(1, 1), name="{}_residual_conv2d".format(self.name))

    def build(self, input_shape):

        # get shape of input matrix x
        self.batch_size = input_shape[0]
        self.num_of_vertices = input_shape[1]
        self.num_of_features = input_shape[2]
        self.num_of_timesteps = input_shape[3]

        super(ASTGCN_block, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        block: time attention --> input, spacial sttention --> cheb_conv_sat --> time_conv + residual_conv
        :param inputs: shape is (batch_size, N, C_{r-1}, T_{r-1})
        :param mask:
        :return: shape is (batch_size, N, num_of_time_filters, T_{r-1})
        """

        # shape is (batch_size, T, T)
        temporal_At = self.TAt(inputs)

        # time attention --> input
        x_TAt = K.reshape(
                K.batch_dot(
                    K.reshape(inputs, (-1, self.num_of_vertices * self.num_of_features, self.num_of_timesteps)),
                    temporal_At),
                (-1, self.num_of_vertices, self.num_of_features, self.num_of_timesteps))
        # cheb gcn with spatial attention
        spatial_At = self.SAt(x_TAt) # batch, N, N
        spatial_gcn = self.cheb_conv_SAt([inputs, spatial_At])  # Batch, N, filter, T
        print('temporal_At', temporal_At.shape, 'spatial_At', spatial_At.shape, 'spatial_gcn', spatial_gcn.shape)
        # convolution along time axis
        # Batch, N, filter, T
        time_conv_output = K.permute_dimensions(self.time_conv(K.permute_dimensions(spatial_gcn, (0, 2, 1, 3))), (0, 2, 1, 3))

        # residual shortcut
        # Batch, N, filter, T
        x_residual = K.permute_dimensions(self.residual_conv(K.permute_dimensions(inputs, (0, 2, 1, 3))), (0, 2, 1, 3))

        res = K.relu(x_residual + time_conv_output)
        print('time_conv_output', time_conv_output.shape, 'x_residual', x_residual.shape, 'res', res.shape)
        return res

    def compute_output_shape(self, input_shape):
        # shape is (batch_size, N, num_of_time_filters, T_{r-1})
        return self.batch_size, self.num_of_vertices, self.num_of_time_filters, self.num_of_timesteps

class ASTGCN_submodule(Layer):

    def __init__(self, num_for_prediction, params, **kwargs):
        super(ASTGCN_submodule, self).__init__(**kwargs)
        self.blocks = []
        for index, param in enumerate(params):
            self.blocks.append(ASTGCN_block(param, name='block_{}'.format(index + 1)))

        self.num_for_prediction = num_for_prediction
        self.final_conv = Conv2D(filters=num_for_prediction,
                                    kernel_size=(1, params[-1]['num_of_time_filters']))

    def build(self, input_shape):

        # get shape of input matrix x
        self.batch_size = input_shape[0]
        self.num_of_vertices = input_shape[1]
        self.num_of_features = input_shape[2]
        self.num_of_timesteps = input_shape[3]

        self.w = self.add_weight(name='w', shape=(self.num_of_vertices, self.num_for_prediction),
                                 initializer='zeros', trainable=True)

        super(ASTGCN_submodule, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        :param inputs: shape is (batch_size, N, C_{r-1}, T_{r-1})
        :param mask:
        :return: shape is (batch_size, num_of_vertices, num_for_prediction)
        """
        x = inputs
        for unit in self.blocks:
            x = unit(x)  # batch, N, filter, T
            print('in {}, {} output shape {}'.format(self.name, unit.name, x.shape))
        module_output = K.permute_dimensions(
            self.final_conv(K.permute_dimensions(x, (0, 3, 1, 2)))[:, :, :, -1], (0, 2, 1))  # batch, N, num_predict
        return module_output * self.w # batch, N, num_predict

    def compute_output_shape(self, input_shape):
        # shape is (batch_size, num_of_vertices, num_for_prediction)
        return self.batch_size, self.num_of_vertices, self.num_for_prediction


def astgcn(args):
    num_of_vertices, num_of_features, num_of_weeks, num_of_days, num_of_hours, cheb_polynomials, num_for_prediction, K =\
        args['num_of_vertices'], args['num_of_features'], args['num_of_weeks'], args['num_of_days'], \
        args['num_of_hours'], args['cheb_polynomials'], args['num_for_prediction'], args['K']

    # params for week, day, hour, three submodules
    # double all the number of [week, day, hour, prediction number], for inflow and outflow
    params1 = [
        {"K": K, "num_of_chev_filters": 64, "num_of_time_filters": 64, "time_conv_strides": num_of_weeks * 2,
         "cheb_polynomials": cheb_polynomials},
        {"K": K, "num_of_chev_filters": 64, "num_of_time_filters": 64, "time_conv_strides": 1 * 2,
         "cheb_polynomials": cheb_polynomials}
    ]
    params2 = [
        {"K": K, "num_of_chev_filters": 64, "num_of_time_filters": 64, "time_conv_strides": num_of_days * 2,
         "cheb_polynomials": cheb_polynomials},
        {"K": K, "num_of_chev_filters": 64, "num_of_time_filters": 64, "time_conv_strides": 1 * 2,
         "cheb_polynomials": cheb_polynomials}
    ]
    params3 = [
        {"K": K, "num_of_chev_filters": 64, "num_of_time_filters": 64, "time_conv_strides": num_of_hours * 2,
         "cheb_polynomials": cheb_polynomials},
        {"K": K, "num_of_chev_filters": 64, "num_of_time_filters": 64, "time_conv_strides": 1 * 2,
         "cheb_polynomials": cheb_polynomials}
    ]
    all_params = [params1, params2, params3]


    hour_input = Input(shape=(num_of_vertices, num_of_features, num_of_hours * 2), dtype='float32', name='hour_input')
    day_input = Input(shape=(num_of_vertices, num_of_features, num_of_days * 2), dtype='float32', name='day_input')
    week_input = Input(shape=(num_of_vertices, num_of_features, num_of_weeks * 2), dtype='float32', name='week_input')
    main_input = [week_input, day_input, hour_input]

    submodules = []
    for params, name in zip(all_params, ['week', 'day', 'hour']):
        submodules.append(ASTGCN_submodule(num_for_prediction * 2, params, name='{}_component'.format(name)))

    submodule_outputs = []
    for idx in range(len(main_input)):
        print('input {} shape: {}'.format(idx, main_input[idx].shape))
        submodule_outputs.append(submodules[idx](main_input[idx]))

    main_output = merge.Add()(submodule_outputs) if len(submodule_outputs) > 1 else submodule_outputs[0]

    model = Model(inputs=main_input, outputs=main_output, name='astgcn')
    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cheb_polynomials = get_cheb_polynomials('/home/ryj/renyajie/exp/GLST_Net/data/taxi_distance.csv', 32 * 32, 3)
    args = {"num_of_vertices": 32 * 32, "num_of_features": 1, "num_of_weeks": 7, "num_of_days": 5, "num_of_hours": 4,
            "cheb_polynomials": cheb_polynomials, "num_for_prediction": 1, "K": 3}
    model = astgcn(args)
    model.summary()