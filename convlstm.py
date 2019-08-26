# -*- coding: utf-8 -*-
"""
main script
"""

# add into system path
import sys
import os

sys.path.append(os.path.join(os.path.curdir, 'preprocess'))
sys.path.append(os.path.join(os.path.curdir, 'model'))

from preprocess import utils
from model import CONVLSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import h5py
import numpy as np
import time
import argparse
import template

exp_name = 'convlstm'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="3", help='use which gpu 0, 1 or 2')
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--len_local', type=int, default=6, help='the length of local time')
parser.add_argument('--neighbor_size', type=int, default=3, help='the size of neighbor size')
parser.add_argument('--stack_convlstm', type=int, default=1, help='how many layer convlstm stack')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
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
data_choice = {}
data_choice['necessary'] = True
data_choice['local_external'] = False
data_choice['local_flow'] = True
data_choice['global_external'] = False
data_choice['global_flow'] = False
param_dict['data_choice'] = data_choice

param_dict['len_global'] = 7
param_dict['len_local'] = len_local
param_dict['neighbor_size'] = neighbor_size

# model params
param_dict['model'] = ['CONVLSTM', 'convlstm',
                       {'len_local': len_local, 'neighbor_size': neighbor_size,
                        'stack_convlstm': args.stack_convlstm}]
param_dict['lr'] = args.lr
param_dict['train'] = True
param_dict['batch_size'] = 32
param_dict['epochs'] = 500

# file params
param_dict['header'] = {'len_local': 10, 'neighbor_size': 15, 'stack_convlstm': 15, 'lr': 8,
                        'start_time': 10, 'end_time': 10, 'time': 20, 'rmse':8, 'mape': 8}
param_dict['hyper_params'] = {'len_local': len_local, 'neighbor_size': neighbor_size,
                             'stack_convlstm': args.stack_convlstm, 'lr': args.lr}

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