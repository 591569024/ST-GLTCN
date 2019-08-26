# -*- coding: utf-8 -*-
"""
main script
"""

# add into system path
import sys
import os

sys.path.append('..')

from preprocess import utils
from model import HA
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import h5py
import numpy as np
import time
import pickle
import argparse
import template

exp_name = 'ha'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
parser.add_argument('--len_local', type=int, default=1, help='the length of local time')
parser.add_argument('--neighbor_size', type=int, default=1, help='the size of neighbor size')
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

len_local = args.len_local  # the length of local time slots
neighbor_size = args.neighbor_size  # the size of neighbor

param_dict = {}

# general param
param_dict['exp_name'] = exp_name
param_dict['is_baseline'] = True
param_dict['dataset'] = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

# data params
data_choice={}
data_choice['necessary'] = True
data_choice['local_external'] = False
data_choice['local_flow'] = True
data_choice['global_external'] = False
data_choice['global_flow'] = False
param_dict['data_choice'] = data_choice

param_dict['len_global'] = 7
param_dict['len_local'] = len_local
param_dict['neighbor_size'] = neighbor_size

# file params
param_dict['header'] = {'len_local': 10, 'neighbor_size': 15, 'start_time': 10, 'end_time': 10,
                        'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'len_local': len_local, 'neighbor_size': neighbor_size}

# model params
param_dict['model'] = ['HA', 'ha', {'len_local': len_local, 'neighbor_size': neighbor_size}]
param_dict['lr'] = 1e-4
param_dict['batch_size'] = 32
param_dict['epochs'] = 100

if __name__ == '__main__':
    # ===============================================================
    # load data
    ts = time.time()
    data, mmn = utils.get_data(param_dict['dataset'], param_dict['len_global'], param_dict['len_local'],
                               param_dict['neighbor_size'], param_dict['data_choice'])

    param_dict['train_data'] = [data[name] for name in ['stack_local_flow_train']]
    param_dict['train_ground'] = data['ground_truth_train']
    param_dict['test_data'] = [data[name] for name in ['stack_local_flow_test']]
    param_dict['test_ground'] = data['ground_truth_test']

    param_dict['max'], param_dict['min'] = mmn._max, mmn._min
    param_dict['start_time'], param_dict['end_time'] = data['predict_time_test'][0], data['predict_time_test'][-1]

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================

    template.run(param_dict)
