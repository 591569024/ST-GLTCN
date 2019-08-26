# -*- coding: utf-8 -*-
"""
main script
"""

# add into system path
import sys
import os

sys.path.append('..')

from preprocess import utils
from model import TCN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import h5py
import numpy as np
import time
import pickle
import argparse
import template

exp_name = 'self_tcn_nog_rdw'
parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
parser.add_argument('--len_recent', type=int, default=4, help='the length of recent time')
parser.add_argument('--len_daily', type=int, default=4, help='the length of daily time')
parser.add_argument('--len_week', type=int, default=4, help='the length of week time')
parser.add_argument('--neighbor_size', type=int, default=2, help='the size of neighbor size')
parser.add_argument('--num_tcn', type=int, default=2, help='how many tcn')
parser.add_argument('--num_cnn', type=int, default=6, help='how many cnn to model current flow')
parser.add_argument('--nb_stacks', type=int, default=2, help='how many stack in a TCN')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--BN', type=int, default=1, help='batch normalization')
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

len_recent = args.len_recent
len_daily = args.len_daily
len_week = args.len_week  # the length of local time slots
neighbor_size = args.neighbor_size  # the size of neighbor

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = False
param_dict['dataset'] = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

# data params
data_choice={}
data_choice['necessary'] = True
data_choice['recent_external'] = True
data_choice['recent_flow'] = True
data_choice['daily_external'] = True
data_choice['daily_flow'] = True
data_choice['week_external'] = True
data_choice['week_flow'] = True
data_choice['current_external'] = True
data_choice['current_flow'] = True
param_dict['data_choice'] = data_choice

param_dict['len_recent'] = len_recent
param_dict['len_daily'] = len_daily
param_dict['len_week'] = len_week
param_dict['neighbor_size'] = neighbor_size

# file params
param_dict['header'] = {'len_recent': 10, 'len_daily': 10, 'len_week': 10, 'neighbor_size': 15, 'num_tcn': 8,
                        'nb_stacks': 10, 'num_cnn': 10, 'lr': 10, 'BN': 8, 'start_time': 10, 'end_time': 10,
                        'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'len_recent': len_recent, 'len_daily': len_daily, 'len_week': len_week,
                              'neighbor_size': neighbor_size, 'num_tcn': args.num_tcn, 'nb_stacks': args.nb_stacks,
                              'num_cnn': args.num_cnn, 'lr': args.lr, 'BN': True if args.BN == 1 else False}

# model params
param_dict['model'] = ['TCN', 'tcn_nog_rdw_att',
                       {'len_recent': len_recent, 'len_daily': len_daily, 'len_week': len_week,
                        'neighbor_size': neighbor_size, 'num_tcn': args.num_tcn, 'nb_stacks': args.nb_stacks,
                        'BN': True if args.BN == 1 else False, 'num_cnn': args.num_cnn,
                        'dataset': param_dict['dataset']}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 32
param_dict['epochs'] = 100

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_rdw_data(param_dict['dataset'], param_dict['len_recent'], param_dict['len_daily'],
                               param_dict['len_week'], param_dict['neighbor_size'], param_dict['data_choice'])

    param_dict['train_data'] = [data[name] for name in ['current_vacation_train', 'current_hour_train', 'current_dayOfWeek_train',
                                     'current_weather_train', 'current_continuous_external_train','recent_vacation_train', 'recent_hour_train', 'recent_dayOfWeek_train',
                                     'recent_weather_train', 'recent_continuous_external_train','daily_vacation_train', 'daily_hour_train', 'daily_dayOfWeek_train',
                                     'daily_weather_train', 'daily_continuous_external_train','week_vacation_train', 'week_hour_train', 'week_dayOfWeek_train',
                                     'week_weather_train', 'week_continuous_external_train', 'current_local_flow_train',
                                     'recent_local_flow_train', 'daily_local_flow_train', 'week_local_flow_train']]
    param_dict['train_ground'] = data['ground_truth_train']
    param_dict['test_data'] = [data[name] for name in ['current_vacation_test', 'current_hour_test', 'current_dayOfWeek_test',
                                     'current_weather_test', 'current_continuous_external_test','recent_vacation_test', 'recent_hour_test', 'recent_dayOfWeek_test',
                                     'recent_weather_test', 'recent_continuous_external_test','daily_vacation_test', 'daily_hour_test', 'daily_dayOfWeek_test',
                                     'daily_weather_test', 'daily_continuous_external_test','week_vacation_test', 'week_hour_test', 'week_dayOfWeek_test',
                                     'week_weather_test', 'week_continuous_external_test', 'current_local_flow_test',
                                     'recent_local_flow_test', 'daily_local_flow_test', 'week_local_flow_test']]
    param_dict['test_ground'] = data['ground_truth_test']

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================

    template.run(param_dict)
