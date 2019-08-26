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
from preprocess import utils
from model import GLSTModel
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import h5py
import numpy as np
import time
import pickle
import argparse
import template
exp_name = 'self_tcn_nog'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="1", help='use which gpu 0, 1 or 2')
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--len_local', type=int, default=3, help='the length of local time')
parser.add_argument('--neighbor_size', type=int, default=1, help='the size of neighbor size')
parser.add_argument('--num_tcn', type=int, default=2, help='how many tcn')
parser.add_argument('--nb_stacks', type=int, default=1, help='how many res unit in a tcn')
parser.add_argument('--num_cnn', type=int, default=6, help='how many cnn')
parser.add_argument('--BN', type=int, default=1, help='whether use batch normalization')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = False
param_dict['dataset'] = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

# data params
data_choice={}
data_choice['necessary'] = True
data_choice['local_external'] = True
data_choice['local_flow'] = True
data_choice['global_external'] = False
data_choice['global_flow'] = False
param_dict['data_choice'] = data_choice

param_dict['len_global'] = 7
param_dict['len_local'] = args.len_local
param_dict['neighbor_size'] = args.neighbor_size

# model params
param_dict['model'] = ['TCN', 'tcn_no_global',
                       {'len_local': args.len_local, 'neighbor_size': args.neighbor_size,
                        'num_tcn': args.num_tcn, 'nb_stacks': args.nb_stacks,
                        'num_cnn': args.num_cnn, 'is_BN': True if args.BN == 1 else False,
                        'dataset': param_dict['dataset']}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 32
param_dict['epochs'] = 500

# file params
param_dict['header'] = {'len_local': 10, 'neighbor_size': 15, 'num_tcn': 10, 'nb_stacks': 10, 'num_cnn': 10, 'is_BN': 10,
                        'lr': 8, 'start_time': 10, 'end_time': 10, 'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'len_local': args.len_local, 'neighbor_size': args.neighbor_size,
                             'num_tcn': args.num_tcn, 'nb_stacks': args.nb_stacks, 'num_cnn': args.num_cnn,
                              'is_BN': True if args.BN == 1 else False, 'lr': args.lr}

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_data(param_dict['dataset'], param_dict['len_global'], param_dict['len_local'],
                               param_dict['neighbor_size'], param_dict['data_choice'])

    if param_dict['dataset'] == 'bj_taxi':
        param_dict['train_data'] = [data[name] for name in ['t_vacation_train', 't_hour_train', 't_dayOfWeek_train',
                                't_weather_train', 't_continuous_external_train', 'current_local_flow_train', 'stack_local_flow_train']]
        param_dict['test_data'] = [data[name] for name in ['t_vacation_test', 't_hour_test', 't_dayOfWeek_test',
                               't_weather_test', 't_continuous_external_test', 'current_local_flow_test', 'stack_local_flow_test']]
    else:
        param_dict['train_data'] = [data[name] for name in ['t_hour_train', 't_dayOfWeek_train',
                                                            'current_local_flow_train', 'stack_local_flow_train']]
        param_dict['test_data'] = [data[name] for name in ['t_hour_test', 't_dayOfWeek_test',
                                                           'current_local_flow_test', 'stack_local_flow_test']]
    param_dict['train_ground'] = data['ground_truth_train']
    param_dict['test_ground'] = data['ground_truth_test']

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    template.run(param_dict)