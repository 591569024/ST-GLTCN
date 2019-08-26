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
exp_name = 'self_nog'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="3", help='use which gpu 0, 1 or 2')
parser.add_argument('--len_local', type=int, default=4, help='the length of local time')
parser.add_argument('--neighbor_size', type=int, default=2, help='the size of neighbor size')
parser.add_argument('--res', type=int, default=12, help='how many res unit')
parser.add_argument('--cnn', type=int, default=6, help='how many cnn unit')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--BN', type=int, default=1, help='whether to use batch normalization or not')
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

len_local = args.len_local  # the length of local time slots
neighbor_size = args.neighbor_size  # the size of neighbor

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = False

# data params
data_choice={}
data_choice['necessary'] = True
data_choice['local_external'] = True
data_choice['local_flow'] = True
data_choice['global_external'] = False
data_choice['global_flow'] = False
param_dict['data_choice'] = data_choice

param_dict['len_global'] = 7
param_dict['len_local'] = len_local
param_dict['neighbor_size'] = neighbor_size

# model params
param_dict['model'] = ['GLSTModel', 'glst_net_no_global',
                       {'len_local': len_local, 'neighbor_size': neighbor_size,
                        'is_BN': True if args.BN == 1 else False, 'nb_cnn': args.cnn}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 32
param_dict['epochs'] = 500

# file params
param_dict['header'] = {'len_local': 10, 'neighbor_size': 15, 'res': 5, 'cnn': 5,
                        'lr': 8, 'BN': 5, 'start_time': 10, 'end_time': 10, 'time': 20, 'rmse':15, 'mape': 15}
param_dict['hyper_params'] = {'len_local': len_local, 'neighbor_size': neighbor_size,
                             'res': args.res, 'cnn': args.cnn, 'lr': args.lr, 'BN': True if args.BN == 1 else False}

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_data(param_dict['len_global'], param_dict['len_local'],
                               param_dict['neighbor_size'], param_dict['data_choice'])

    param_dict['train_data'] = [data[name] for name in ['t_vacation_train', 't_hour_train', 't_dayOfWeek_train',
                            't_weather_train', 't_continuous_external_train', 'current_local_flow_train',
                            'stack_local_flow_train']]
    param_dict['train_ground'] = data['ground_truth_train']
    param_dict['test_data'] = [data[name] for name in ['t_vacation_test', 't_hour_test', 't_dayOfWeek_test',
                           't_weather_test', 't_continuous_external_test', 'current_local_flow_test',
                           'stack_local_flow_test']]
    param_dict['test_ground'] = data['ground_truth_test']

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    template.run(param_dict)