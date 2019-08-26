# -*- coding: utf-8 -*-
"""
main script
"""

# add into system path
import sys
import os

sys.path.append(os.path.join(os.path.curdir, 'preprocess'))
sys.path.append(os.path.join(os.path.curdir, 'model'))

from preprocess import utils, LoadData
from model import ASTGCN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import h5py
import numpy as np
import time
import argparse
import template
import math
from preprocess.LoadData import MinMaxNormalization

exp_name = 'astgcn'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--neighbor_size', type=int, default=1, help='the size of neighbor size')
parser.add_argument('--K', type=int, default=3, help='how many neighborhoods for a single node')
parser.add_argument('--num_of_features', type=int, default=1, help='how many feature, just 1 for flow')
parser.add_argument('--num_for_prediction', type=int, default=1, help='predict length')
parser.add_argument('--points_per_hour', type=int, default=2, help='how many point in an hour')
parser.add_argument('--num_of_weeks', type=int, default=4, help='how many week len as historical data')
parser.add_argument('--num_of_days', type=int, default=2, help='how many day as historical data')
parser.add_argument('--num_of_hours', type=int, default=4, help='how many hour as historical data')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

len_local = 4  # the length of local time slots
neighbor_size = args.neighbor_size  # the size of neighbor

num_of_vertices = 32 * 32 if args.dataset == 'taxi' else 16 * 8
distance_files = {'bike': '/home/ryj/renyajie/exp/GLST_Net/data/bike_distance.csv',
                  'taxi': '/home/ryj/renyajie/exp/GLST_Net/data/taxi_distance.csv'}
distance_file = distance_files[args.dataset]

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = True
param_dict['dataset'] = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

# data params
data_choice = {}
data_choice['necessary'] = True
data_choice['local_external'] = False
data_choice['local_flow'] = True
data_choice['global_external'] = False
data_choice['global_flow'] = False
param_dict['data_choice'] = data_choice

param_dict['len_global'] = 7
param_dict['len_local'] = 4
param_dict['neighbor_size'] = neighbor_size

# model params
param_dict['model'] = ['ASTGCN', 'astgcn',
                       {'num_of_vertices': num_of_vertices, 'num_of_features': args.num_of_features,
                        'num_of_weeks': args.num_of_weeks, "num_of_days": args.num_of_days,
                        'num_of_hours': args.num_of_hours,
                        'num_for_prediction': args.num_for_prediction,
                        'K': args.K,
                        'cheb_polynomials': ASTGCN.get_cheb_polynomials(distance_file, num_of_vertices, args.K)}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 32
param_dict['epochs'] = 5000

# file params
param_dict['header'] = {'neighbor_size': 15, 'vertices': 10,
                        'features': 10, 'weeks': 8, 'days': 6, 'hours': 6, 'lr': 8,
                        'start_time': 10, 'end_time': 10, 'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'neighbor_size': neighbor_size,
                             'vertices': num_of_vertices, 'features': args.num_of_features,
                              'weeks': args.num_of_weeks, 'days': args.num_of_days, 'hours': args.num_of_hours,
                              'lr': args.lr}
# ===============================================================
unit_day = 48 if param_dict['dataset'] == 'bj_taxi' else 24
proportion_test = 0.1

def get_data(dataset, predict_time, hours, days, weeks):
    """
    get the rest flow data, default the
    :param predict_time:
    :param neighbor_size: the size of map = 2*value + 1
    :param index_cut: the position to cut
    :param hours：
    :param days:
    :param weeks:
    :return:
     hours(None, vertices, feature, hours * 2), days(None, vertices, feature, days * 2),
     weeks(None, vertices, feature, weeks * 2), ground_truth(None, vertices, feature, predict_len * 2)
    """

    X = []
    Y = []
    T = []

    if dataset == 'bj_taxi':
        complete_date, complete_data, _, date_index = LoadData.load_bj_taxi_flow()
    else:
        complete_date, complete_data, _, date_index = LoadData.load_ny_bike_flow()

    if dataset == 'bj_taxi':
        row, column = 32, 32
    else:
        row, column = 16, 8

    for i, index in enumerate(date_index):

        hour_end, hour_start = i - 1, i - hours
        day_end, day_start = i - unit_day, i - unit_day * days
        week_end, week_start = i - unit_day * 7, i - unit_day * 7 * weeks

        if hour_start >= 0 and hour_end >= 0 \
                and day_start >= 0 and day_end >= 0 \
                and week_start >= 0 and week_end >= 0:
            # shape: (time_step * 2, row, column)
            hour = complete_data[hour_start : hour_end + 1 : 1, :, :, :]
            day = complete_data[day_start : day_end + 1 : unit_day, :, :, :]
            week = complete_data[week_start : week_end + 1 : unit_day * 7, :, :, :]
            ground = complete_data[i, :, :, :]

            # (batch, time_step * 2, row, column) --> (batch, vertices, feature, time_steps * 2)
            hour = hour.reshape((-1, row * column))
            hour = np.expand_dims(np.transpose(hour, (1, 0)), axis=-2)
            day = day.reshape((-1, row * column))
            day = np.expand_dims(np.transpose(day, (1, 0)), axis=-2)
            week = week.reshape((-1, row * column))
            week = np.expand_dims(np.transpose(week, (1, 0)), axis=-2)

            # ground truth, (batch, vertices, prediction * 2)
            ground = ground.reshape((-1, row * column))
            ground = np.transpose(ground, (1, 0))

            X.append(np.concatenate([hour, day, week], axis=-1))
            Y.append(ground)
            T.append(index)

    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)

    # cut into test and train data
    total_length = X.shape[0]
    len_test = math.ceil(total_length * proportion_test)

    X_train, X_test = X[:-len_test], X[-len_test:]
    Y_train, Y_test = Y[:-len_test], Y[-len_test:]
    T_train, T_test = T[:-len_test], T[-len_test:]

    return X_train, X_test, Y_train, Y_test, T_train, T_test

def quick_get_data(dataset, predict_time, hours, days, weeks):
    """
    load astgcn data quickly if there is the file
    :param predict_time:
    :param index_cut:
    :param neighbor_size:
    :param hours:
    :param days:
    :param weeks:
    :param predict_len: 1 default
    :return:
    """
    start_time = str(predict_time[0])
    end_time = str(predict_time[-1])
    path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/data'

    filename = '{}_baseline_astgcn_{}_{}_{}.h5'.format(dataset, hours, days, weeks)
    f = h5py.File(os.path.join(path, filename), 'a')

    # encode the time
    encode_time = np.asarray([utils.encode_time(batch) for batch in predict_time])

    # if the same, load directly
    if 'predict_time' in f and (encode_time == f['predict_time'][()]).all():
        print('cache load astgcn {} data from {} to {}'.format(dataset, start_time, end_time))
        print('-' * 30)

        X_train = f['X_train'][()]
        X_test = f['X_test'][()]
        Y_train = f['Y_train'][()]
        Y_test = f['Y_test'][()]
        T_train = f['T_train'][()]
        T_test = f['T_test'][()]

        f.close()
        return X_train, X_test, Y_train, Y_test, T_train, T_test

    f.close()

    # else calculate then cache
    f = h5py.File(os.path.join(path, filename), 'w')

    print('calculate astgcn {} data from {} to {}'.format(dataset, start_time, end_time))

    X_train, X_test, Y_train, Y_test, T_train, T_test = \
        get_data(dataset, predict_time, hours, days, weeks)

    encode_T_train = np.asarray([utils.encode_time(batch) for batch in T_train])
    encode_T_test = np.asarray([utils.encode_time(batch) for batch in T_test])

    print('cache astgcn {} data from {} to {}'.format(dataset, start_time, end_time))

    f['predict_time'] = encode_time
    f['X_train'] = X_train
    f['X_test'] = X_test
    f['Y_train'] = Y_train
    f['Y_test'] = Y_test
    f['T_train'] = encode_T_train
    f['T_test'] = encode_T_test

    f.close()
    return X_train, X_test, Y_train, Y_test, T_train, T_test

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_data(param_dict['dataset'], param_dict['len_global'], param_dict['len_local'],
                               param_dict['neighbor_size'], param_dict['data_choice'])

    predict_time = np.concatenate([data['predict_time_train'], data['predict_time_test']])
    index_cut = np.concatenate([data['index_cut_train'], data['index_cut_test']])

    X_train, X_test, Y_train, Y_test, T_train, T_test = \
        quick_get_data(param_dict['dataset'], predict_time, args.num_of_hours,
                       args.num_of_days, args.num_of_weeks)

    train_hour, train_day, train_week = \
        X_train[:, :, :, :args.num_of_hours * 2], \
        X_train[:, :, :, args.num_of_hours * 2:-args.num_of_weeks * 2], \
        X_train[:, :, :, -args.num_of_weeks * 2:]

    test_hour, test_day, test_week = \
        X_test[:, :, :, :args.num_of_hours * 2], \
        X_test[:, :, :, args.num_of_hours * 2:-args.num_of_weeks * 2], \
        X_test[:, :, :, -args.num_of_weeks * 2:]



    # create data [week, day, hour] (batch, vertices, feature, time_steps)
    param_dict['train_data'] = [train_week, train_day, train_hour]
    param_dict['train_ground'] = Y_train
    param_dict['test_data'] = [test_week, test_day, test_hour]
    param_dict['test_ground'] = Y_test

    print(X_train.shape)
    for i in param_dict['train_data']:
        print(i.shape)

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    template.run(param_dict)