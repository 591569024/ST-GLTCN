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
from model import STDN
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

exp_name = 'stdn'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="1", help='use which gpu 0, 1 or 2')
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--att_lstm_num', type=int, default=3, help='the length of close time')
parser.add_argument('--att_lstm_seq_len', type=int, default=7, help='the length of trend time')
parser.add_argument('--lstm_seq_len', type=int, default=7, help='the length of period time')
parser.add_argument('--flow_gate_len', type=int, default=2, help='the length of period time')
parser.add_argument('--neighbor_size', type=int, default=3, help='the size of neighbor size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = False
param_dict['dataset'] = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

# data params
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
param_dict['model'] = ['STDN', 'stdn',
                       {'att_lstm_num': args.att_lstm_num, 'att_lstm_seq_len': args.att_lstm_seq_len,
                        'lstm_seq_len': args.lstm_seq_len, 'neighbor_size': args.neighbor_size}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 64
param_dict['epochs'] = 500

# file params
param_dict['header'] = {'att_lstm_num': 15, 'att_lstm_seq_len': 20, 'lstm_seq_len': 15, 'flow_gate_len': 15,
                        'neighbor_size': 15, 'start_time': 10, 'end_time': 10, 'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'att_lstm_num': args.att_lstm_num, 'att_lstm_seq_len': args.att_lstm_seq_len,
                              'lstm_seq_len': args.lstm_seq_len, 'flow_gate_len': args.flow_gate_len,
                              'neighbor_size': args.neighbor_size}

# ===============================================================

unit_day = 48 if param_dict['dataset'] == 'bj_taxi' else 24
proportion_test = 0.1

data_name = ['lstm_input', 'flow_input', 'nbhd_input', 'att_nbhd_input',
             'att_flow_input', 'att_lstm_input', 'ground_truth']

if args.att_lstm_seq_len % 2 != 1:
    print("att_lstm_seq_len should be an odd number")
    exit(1)

def load_bj_taxi_holiday(timeslots):
    """
    ret:  holiday list
    """
    path = os.path.join(utils.get_data_path(), 'TaxiBJ', 'BJ_Holiday.txt')

    with open(path, 'r') as f:
        holiday = f.readlines()
        holiday = list(map(str.strip, holiday))

    # 将预测日期中为节假日的日期置为1
    feature_holiday = []

    for time in timeslots:
        if time[:-2] in holiday:
            feature_holiday.append(1)
        else:
            feature_holiday.append(0)

    return np.asarray(feature_holiday)

def load_bj_taxi_meteorology(timeslots=None):
    path = os.path.join(utils.get_data_path(), 'TaxiBJ', 'BJ_Meteorology.h5')

    with h5py.File(path, 'r+') as f:
        Weather = f['Weather'][()]
        date = f['date'][()]

    # byte->str
    date = np.array([time.decode('utf-8') for time in date])

    # 创建索引
    index = [np.where(date == time)[0][0] for time in timeslots]

    weather = []

    for idx in index:
        weather.append(Weather[idx - 1])
    weather = np.asarray(weather)
    return weather

def load_ny_bike_external(timeslots):

    date = [time[:-2] for time in timeslots]
    hour = [int(time[-2:]) for time in timeslots]
    dayOfWeek = [datetime.strptime(time, "%Y%m%d").weekday() for time in date]

    hour = np.asarray(hour)
    dayOfWeek = np.asarray(dayOfWeek)

    return hour, dayOfWeek

def get_data(dataset, predict_time, index_cut, neighbor_size, lstm_seq_len, flow_gate_len, att_lstm_num, att_lstm_seq_len):
    """

    :param predict_time:
    :param index_cut:
    :param neighbor_size:
    :param lstm_seq_len: short lstm length
    :param att_lstm_num: the number of attention
    :param att_lstm_seq_len: the scope of interval each attention contains
    :return:
    """

    slide_length = neighbor_size * 2 + 1

    if dataset == 'bj_taxi':
        complete_date, complete_data, _, date_index = LoadData.load_bj_taxi_flow()
    else:
        complete_date, complete_data, _, date_index = LoadData.load_ny_bike_flow()

    total_length = len(date_index)

    lstm_input = []
    flow_input = []
    nbhd_input = []
    att_nbhd_input = []
    att_flow_input = []
    att_lstm_input = []
    ground_truth = []

    # lstm_input
    for index, (row_offset, column_offset) in enumerate(index_cut):
        i = date_index[predict_time[index]]

        # check the bounnd
        bound_att_1, bound_att_2 = i - 1 - flow_gate_len - att_lstm_seq_len // 2 * unit_day, \
                                   i - 1 + att_lstm_seq_len // 2 * unit_day
        bound_local_1, bound_local_2 = i - lstm_seq_len - flow_gate_len, i - 1

        if not (bound_att_1 >= 0 and bound_att_1 < total_length and \
            bound_att_2 >= 0 and bound_att_2 < total_length and \
            bound_local_1 >= 0 and bound_local_2 < total_length and \
            bound_local_2 >= 0 and bound_local_2 < total_length):
            continue

        row_end = row_offset + slide_length
        column_end = column_offset + slide_length

        ground_truth.append(complete_data[i, :, row_offset:row_end, column_offset:column_end])

        # short
        temp_nbhd_input = []
        temp_flow_input = []
        local_time = []

        for short_index in range(lstm_seq_len, 0, -1):
            local_current = i - short_index
            flow_current_start, flow_current_end = i - short_index - flow_gate_len, i - short_index - 1

            nbhd_input_ = complete_data[local_current, :, row_offset:row_end, column_offset:column_end]
            flow_input_ = np.zeros(shape=(2, slide_length, slide_length))
            for one_day in complete_data[flow_current_start:flow_current_end + 1, :, row_offset:row_end, column_offset:column_end]:
                flow_input_ = flow_input_ + one_day

            temp_nbhd_input.append(nbhd_input_)
            temp_flow_input.append(flow_input_)
            local_time.append(complete_date[local_current])

        if dataset == 'bj_taxi':
            short_meteology, _, _ = LoadData.load_bj_taxi_meteorology(local_time)
            vacation, hour, dayOfWeek = LoadData.load_other_bj_taxi_external(local_time)
            temp_lstm_input = np.hstack([short_meteology[:, None], vacation[:, None], hour[:, None], dayOfWeek[:, None]])
        else:
            hour_list, dayOfWeek_list = load_ny_bike_external(local_time)
            temp_lstm_input = np.hstack([hour_list[:, None], dayOfWeek_list[:, None]])

        lstm_input.append(temp_lstm_input)
        flow_input.append(temp_flow_input)
        nbhd_input.append(temp_nbhd_input)

        # long
        temp_att_nbhd_input = []
        temp_att_flow_input = []
        temp_att_lstm_input = []

        for att in range(att_lstm_num, 0, -1):
            long_time = []
            for seq in range(att_lstm_seq_len // 2 * -1, att_lstm_seq_len // 2 + 1):

                att_current = i - 1 + seq - att * unit_day
                att_flow_start, att_flow_end = att_current - flow_gate_len, att_current - 1

                att_nbhd_input_ = complete_data[att_current, :, row_offset:row_end, column_offset:column_end]
                att_flow_input_ = np.zeros(shape=(2, slide_length, slide_length))

                for one_day in complete_data[att_flow_start:att_flow_end+1, :, row_offset:row_end, column_offset:column_end]:
                    att_flow_input_ = att_flow_input_ + one_day

                temp_att_nbhd_input.append(att_nbhd_input_)
                temp_att_flow_input.append(att_flow_input_)
                long_time.append(complete_date[att_current])

            if dataset == 'bj_taxi':
                long_meteology, _, _ = LoadData.load_bj_taxi_meteorology(long_time)
                vacation, hour, dayOfWeek = LoadData.load_other_bj_taxi_external(long_time)
                temp_att_lstm_input_ = np.hstack([long_meteology[:, None], vacation[:, None], hour[:, None], dayOfWeek[:, None]])
            else:
                hour_list, dayOfWeek_list = load_ny_bike_external(long_time)
                temp_att_lstm_input_ = np.hstack([hour_list[:, None], dayOfWeek_list[:, None]])

            temp_att_lstm_input.append(temp_att_lstm_input_)

        att_lstm_input.append(temp_att_lstm_input)
        att_nbhd_input.append(temp_att_nbhd_input)
        att_flow_input.append(temp_att_flow_input)

    data_dict = {}
    data_dict['lstm_input'] = np.array(lstm_input)
    data_dict['flow_input'] = np.array(flow_input)
    data_dict['nbhd_input'] = np.array(nbhd_input)
    data_dict['att_nbhd_input'] = np.array(att_nbhd_input)
    data_dict['att_flow_input'] = np.array(att_flow_input)
    data_dict['att_lstm_input'] = np.array(att_lstm_input)
    data_dict['ground_truth'] = np.array(ground_truth)

    # cut into test and train data
    total_length = data_dict['att_lstm_input'].shape[0]
    len_test = math.ceil(total_length * proportion_test)

    test_dict = {key + '_test': value[-len_test:] for key, value in data_dict.items()}
    train_dict = {key + '_train': value[:-len_test] for key, value in data_dict.items()}

    external_dim = data_dict['att_lstm_input'].shape[-1]

    return train_dict, test_dict, external_dim

def quick_get_data(dataset, predict_time, index_cut, neighbor_size, lstm_seq_len, flow_gate_len, att_lstm_num, att_lstm_seq_len):

    start_time = str(predict_time[0])
    end_time = str(predict_time[-1])
    path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/data'

    filename = '{}_baseline_stdn_neighbor_{}_{}_{}_{}.h5'.format(dataset, neighbor_size, lstm_seq_len, flow_gate_len, att_lstm_num, att_lstm_seq_len)
    f = h5py.File(os.path.join(path, filename), 'a')

    # encode the time
    encode_time = np.asarray([utils.encode_time(batch) for batch in predict_time])

    # if the same, load directly
    if 'predict_time' in f and (encode_time == f['predict_time'][()]).all():
        print('cache load stdn {} data from {} to {}, neighbor is {}'.format(dataset, start_time, end_time, neighbor_size))
        print('-' * 30)

        train_dict = {key + '_train' : f[key + '_train'][()] for key in data_name}
        test_dict = {key + '_test' : f[key + '_test'][()] for key in data_name}
        external_dim = f['external_dim'][()]

        if(train_dict['lstm_input_train'].shape[1] == lstm_seq_len
                and train_dict['flow_input_train'].shape[3] == neighbor_size * 2 + 1
                and train_dict['att_nbhd_input_train'].shape[1] == att_lstm_num * att_lstm_seq_len
                and train_dict['att_lstm_input_train'].shape[1] == att_lstm_num):
            f.close()
            return train_dict, test_dict, external_dim

    f.close()

    # else calculate then cache
    f = h5py.File(os.path.join(path, filename), 'w')

    print('calculate stdn {} data from {} to {}, neighbor is {}'.format(dataset, start_time, end_time, neighbor_size))
    train_dict, test_dict, external_dim = \
        get_data(dataset, predict_time, index_cut, neighbor_size, lstm_seq_len, flow_gate_len, att_lstm_num, att_lstm_seq_len)

    print('cache stdn {} data from {} to {}, neighbor is {}'.format(dataset, start_time, end_time, neighbor_size))

    f['predict_time'] = encode_time

    for data in [test_dict, train_dict]:
        for key, value in data.items():
            f[key] = value

    f['external_dim'] = external_dim

    f.close()
    return train_dict, test_dict, external_dim

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_data(param_dict['dataset'], param_dict['len_global'], param_dict['len_local'],
                               param_dict['neighbor_size'], param_dict['data_choice'])

    predict_time = np.concatenate([data['predict_time_train'], data['predict_time_test']])
    index_cut = np.concatenate([data['index_cut_train'], data['index_cut_test']])
    train_dict, test_dict, external_dim = \
        quick_get_data(param_dict['dataset'], predict_time, index_cut, args.neighbor_size,
                       args.lstm_seq_len, args.flow_gate_len, args.att_lstm_num, args.att_lstm_seq_len)

    for key, value in train_dict.items():
        print(key, value.shape)

    # change the order of batch and unit size
    train_inputs = \
        train_dict['att_nbhd_input_train'].transpose(1, 0, 2, 3, 4).tolist() + \
        train_dict['att_flow_input_train'].transpose(1, 0, 2, 3, 4).tolist() + \
        train_dict['att_lstm_input_train'].transpose(1, 0, 2, 3).tolist() + \
        train_dict['nbhd_input_train'].transpose(1, 0, 2, 3, 4).tolist() + \
        train_dict['flow_input_train'].transpose(1, 0, 2, 3, 4).tolist() + \
        [train_dict['lstm_input_train'].tolist()]

    test_inputs = \
        test_dict['att_nbhd_input_test'].transpose(1, 0, 2, 3, 4).tolist() + \
        test_dict['att_flow_input_test'].transpose(1, 0, 2, 3, 4).tolist() + \
        test_dict['att_lstm_input_test'].transpose(1, 0, 2, 3).tolist() + \
        test_dict['nbhd_input_test'].transpose(1, 0, 2, 3, 4).tolist() + \
        test_dict['flow_input_test'].transpose(1, 0, 2, 3, 4).tolist() + \
        [test_dict['lstm_input_test'].tolist()]

    param_dict['train_data'] = train_inputs
    param_dict['train_ground'] = train_dict['ground_truth_train']
    param_dict['test_data'] = test_inputs
    param_dict['test_ground'] = test_dict['ground_truth_test']

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    # special, add one model parameter for different external size, due to different dataset
    param_dict['model'][2]['external_dim'] = external_dim

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    template.run(param_dict)