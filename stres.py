# -*- coding: utf-8 -*-
"""
main script
"""

# add into system path
import sys
import os

sys.path.append(os.path.join(os.path.curdir, 'preprocess'))
sys.path.append(os.path.join(os.path.curdir, 'model'))

from preprocess import LoadData
from preprocess import utils, LoadData
from model import ST_Res
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import h5py
import numpy as np
import time
import pickle
import math
import argparse
import template
from datetime import datetime
from preprocess.LoadData import MinMaxNormalization

exp_name = 'stres'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="3", help='use which gpu 0, 1 or 2')
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--len_close', type=int, default=3, help='the length of close time')
parser.add_argument('--len_trend', type=int, default=2, help='the length of trend time')
parser.add_argument('--len_period', type=int, default=2, help='the length of period time')
parser.add_argument('--neighbor_size', type=int, default=1, help='the size of neighbor size')
parser.add_argument('--res', type=int, default=12, help='the number of res unit')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = False
param_dict['dataset'] = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

data_choice={}
data_choice['necessary'] = True
data_choice['local_external'] = False
data_choice['local_flow'] = False
data_choice['global_external'] = False
data_choice['global_flow'] = False
param_dict['data_choice'] = data_choice

param_dict['len_global'] = 7
param_dict['len_local'] = 4
param_dict['neighbor_size'] = args.neighbor_size

# model params
param_dict['model'] = ['ST_Res', 'resnet',
                       {'neighbor_size': args.neighbor_size, 'len_close': args.len_close,
                        'len_period': args.len_period, 'len_trend': args.len_trend, 'nb_residual_unit': args.res}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 32
param_dict['epochs'] = 500

# file params
param_dict['header'] = {'len_close': 15, 'len_trend': 20, 'len_period': 15, 'neighbor_size': 15,
                        'res': 15, 'lr': 10, 'start_time': 10, 'end_time': 10, 'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'len_close': args.len_close, 'len_trend': args.len_trend,
                              'len_period': args.len_period, 'neighbor_size': args.neighbor_size,
                              'res': args.res, 'lr': args.lr}

# ===============================================================
unit_day = 48 if param_dict['dataset'] == 'bj_taxi' else 24
proportion_test = 0.1

def load_ny_bike_external(timeslots, description):

    date = [time[:-2] for time in timeslots]
    hour = [int(time[-2:]) for time in timeslots]
    dayOfWeek = [datetime.strptime(time, "%Y%m%d").weekday() for time in date]

    hour = np.asarray(hour)
    dayOfWeek = np.asarray(dayOfWeek)

    # todo remove this, for print
    print('stres special calculating {}'.format(description),
          '\n' + '-' * 30,
          '\nhour:', hour.shape,
          '\ndayOfWeek:', dayOfWeek.shape,
          '\n' + '-' * 30)

    return hour, dayOfWeek

def get_data(dataset, predict_time, index_cut, neighbor_size, len_close, len_period, len_trend):
    """
    get the rest flow data, default the
    :param predict_time:
    :param neighbor_size: the size of map = 2*value + 1
    :param index_cut: the position to cut
    :param res_len_close：
    :param res_len_trend:
    :param res_len_period:
    :return:
     close(None, 2 * close, slide, slide), trend(None, 2 * trend, slide, slide),
     period(None, 2 * period, slide, slide), ground_truth(None, 2, slide, slide)
     external(None, 20)
    """
    slide_length = neighbor_size * 2 + 1

    X = []
    Y = []
    T = []

    if dataset == 'bj_taxi':
        complete_date, complete_data, _, date_index = LoadData.load_bj_taxi_flow()
    else:
        complete_date, complete_data, _, date_index = LoadData.load_ny_bike_flow()

    # todo remove
    for index, (row_offset, column_offset) in enumerate(index_cut):
        i = date_index[predict_time[index]]

        close_end, close_start = i - 1, i - len_close
        period_end, period_start = i - unit_day, i - unit_day * len_period
        trend_end, trend_start = i - unit_day * 7, i - unit_day * 7 * len_trend

        row_end = row_offset + slide_length
        column_end = column_offset + slide_length

        if close_start >= 0 and close_end >= 0 \
                and period_start >= 0 and period_end >= 0 \
                and trend_start >= 0 and trend_end >= 0:

            # shape: [time_slots, 2, row, column]
            close = complete_data[close_start: close_end + 1 : 1, :, row_offset:row_end, column_offset:column_end]
            period = complete_data[period_start: period_end + 1 : unit_day, :, row_offset:row_end, column_offset:column_end]
            trend = complete_data[trend_start: trend_end + 1 : unit_day * 7, :, row_offset:row_end, column_offset:column_end]

            # shape: [time_slots * 2, row, column]
            # (batch, time_step * 2, row, column) --> (batch, vertices, feature, time_steps)
            close = np.vstack(close)
            period = np.vstack(period)
            trend = np.vstack(trend)

            # shape: [(hour + day + week) * 2, row, column]
            X.append(np.vstack([close, period, trend]))
            Y.append(complete_data[i, :, row_offset:row_end, column_offset:column_end])
            # 保存要预测的时间
            T.append(predict_time[index])

    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)

    # deal with the external data
    if dataset == 'bj_taxi':
        # todo fix--can't use LoadData(global(batch, [1~7]))
        # vacation_list, hour_list, dayOfWeek_list, weather_list, continuous_external_list = \
        #    LoadData.load_bj_taxi_external(T, 'bj taxi stres baseline')
        # External = np.hstack([vacation_list[:,None], hour_list[:,None], dayOfWeek_list[:,None],
        #                       weather_list[:,None], continuous_external_list])
        weather, temperature, windSpeed = LoadData.load_bj_taxi_meteorology(T)
        vacation, hour, dayOfWeek = LoadData.load_other_bj_taxi_external(T)
        External = np.hstack([vacation[:, None], hour[:, None], dayOfWeek[:, None], weather[:, None],
                              temperature[:, None], windSpeed[:, None]])
    else:
        # todo fix--can't use LoadData(global(batch, [1~7]))
        # hour_list, dayOfWeek_list = LoadData.load_ny_bike_external(T, 'ny bike stres baseline')
        # External = np.hstack([hour_list[:, None], dayOfWeek_list[:, None]])
        hour_list, dayOfWeek_list = load_ny_bike_external(T, 'load ny bike external')
        External = np.hstack([hour_list[:, None], dayOfWeek_list[:, None]])

    # cut into test and train data
    total_length = External.shape[0]
    len_test = math.ceil(total_length * proportion_test)

    X_train, X_test = X[:-len_test], X[-len_test:]
    Y_train, Y_test = Y[:-len_test], Y[-len_test:]
    T_train, T_test = T[:-len_test], T[-len_test:]
    External_train, External_test = External[:-len_test], External[-len_test:]
    external_dim = External.shape[-1]

    return X_train, X_test, Y_train, Y_test, T_train, T_test, External_train, External_test, external_dim

def quick_get_data(dataset, predict_time, index_cut, neighbor_size, len_close, len_period, len_trend):
    """
    load st-res data quickly if there is the file
    :param predict_time:
    :param index_cut:
    :param neighbor_size:
    :param len_close:
    :param len_period:
    :param len_trend:
    :return:
    """
    start_time = str(predict_time[0])
    end_time = str(predict_time[-1])
    path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/data'

    filename = '{}_baseline_stres_neighbor_{}_{}_{}_{}.h5'.format(dataset, neighbor_size, len_close, len_period, len_trend)
    f = h5py.File(os.path.join(path, filename), 'a')

    # encode the time
    encode_time = np.asarray([utils.encode_time(batch) for batch in predict_time])

    # if the same, load directly
    if 'predict_time' in f and (encode_time == f['predict_time'][()]).all():
        print('cache load st-res {} data from {} to {}, neighbor is {}'.format(dataset, start_time, end_time, neighbor_size))
        print('-' * 30)

        X_train = f['X_train'][()]
        X_test = f['X_test'][()]
        Y_train = f['Y_train'][()]
        Y_test = f['Y_test'][()]
        T_train = f['T_train'][()]
        T_test = f['T_test'][()]
        External_train = f['External_train'][()]
        External_test = f['External_test'][()]
        external_dim = f['external_dim'][()]

        f.close()
        return X_train, X_test, Y_train, Y_test, T_train, T_test, External_train, External_test, external_dim

    f.close()

    # else calculate then cache
    f = h5py.File(os.path.join(path, filename), 'w')

    print('calculate st-res {} data from {} to {}, neighbor is {}'.format(dataset, start_time, end_time, neighbor_size))

    X_train, X_test, Y_train, Y_test, T_train, T_test, External_train, External_test, external_dim = \
        get_data(dataset, predict_time, index_cut, neighbor_size, len_close, len_period, len_trend)

    encode_T_train = np.asarray([utils.encode_time(batch) for batch in T_train])
    encode_T_test = np.asarray([utils.encode_time(batch) for batch in T_test])

    print('cache st-res {} data from {} to {}, neighbor is {}'.format(dataset, start_time, end_time, neighbor_size))

    f['predict_time'] = encode_time
    f['X_train'] = X_train
    f['X_test'] = X_test
    f['Y_train'] = Y_train
    f['Y_test'] = Y_test
    f['T_train'] = encode_T_train
    f['T_test'] = encode_T_test
    f['External_train'] = External_train
    f['External_test'] = External_test
    f['external_dim'] = external_dim

    f.close()
    return X_train, X_test, Y_train, Y_test, T_train, T_test, External_train, External_test, external_dim

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_data(param_dict['dataset'], param_dict['len_global'], param_dict['len_local'],
                               param_dict['neighbor_size'], param_dict['data_choice'])

    predict_time = np.concatenate([data['predict_time_train'], data['predict_time_test']])
    index_cut = np.concatenate([data['index_cut_train'], data['index_cut_test']])

    X_train, X_test, Y_train, Y_test, T_train, T_test, External_train, External_test, external_dim = \
        quick_get_data(param_dict['dataset'], predict_time, index_cut, args.neighbor_size, args.len_close,
                       args.len_period, args.len_trend)

    train_close, train_period, train_trend = \
        X_train[:, :args.len_close * 2], \
        X_train[:, args.len_close * 2:-args.len_trend * 2], \
        X_train[:, -args.len_trend * 2:]

    test_close, test_period, test_trend = \
        X_test[:, :args.len_close * 2], \
        X_test[:, args.len_close * 2:-args.len_trend * 2], \
        X_test[:, -args.len_trend * 2:]

    param_dict['train_data'] = [train_close, train_period, train_trend, External_train]
    param_dict['train_ground'] = Y_train
    param_dict['test_data'] = [test_close, test_period, test_trend, External_test]
    param_dict['test_ground'] = Y_test

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    # special, add one model parameter for different external size, due to different dataset
    param_dict['model'][2]['external_dim'] = external_dim

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    template.run(param_dict)
