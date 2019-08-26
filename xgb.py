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
from model import XGBOOST
import h5py
import numpy as np
import time
import pickle
import argparse

from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.model_selection import train_test_split

exp_name = 'xgb'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--gpu', type=str, default="1", help='use which gpu 0, 1 or 2')
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--len_local', type=int, default=3, help='len_local')
parser.add_argument('--neighbor_size', type=int, default=2, help='neighbor_size')
parser.add_argument('--deep', type=int, default=1800, help='neighbor_size')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

len_global = 7  # the length of global time slots
len_local = args.len_local  # the length of local time slots
neighbor_size = args.neighbor_size  # the size of neighbor

#seq_len = args.seq_len
#region_len = args.region_len

data_choice={}
data_choice['necessary'] = True
data_choice['local_external'] = False
data_choice['local_flow'] = False
data_choice['global_external'] = False
data_choice['global_flow'] = False
dataset = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

def get_data(predict_time, neighbor_size, index_cut):
    """
    get the arima data
    :param predict_time:
    :param neighbor_size:
    :param seq_len: how many local data contains
    :param index_cut:
    :return:
    """
    slide_length = neighbor_size * 2 + 1

    xgboost_inflow_data = []
    xgboost_outflow_data = []
    xgboost_inflow_ground_truth = []
    xgboost_outflow_ground_truth = []

    complete_date, complete_data, _, date_index = LoadData.load_bj_taxi_flow() \
        if dataset == 'bj_taxi' else LoadData.load_ny_bike_flow()

    for index, (row_offset, column_offset) in enumerate(index_cut):
        end_index = date_index[predict_time[index]]
        for row in range(slide_length):
            for column in range(slide_length):
                inflow_ground = complete_data[end_index, 0, row + row_offset, column + column_offset]
                outflow_ground = complete_data[end_index, 1, row + row_offset, column + column_offset]

                xgboost_inflow_data.append(
                    complete_data[end_index-len_local:end_index, 0, row+row_offset, column+column_offset])
                xgboost_outflow_data.append(
                    complete_data[end_index-len_local:end_index, 1, row+row_offset, column+column_offset])

                # BUG: GammaRegression: label must be nonnegative, so add 1
                xgboost_inflow_ground_truth.append(inflow_ground + 1)
                xgboost_outflow_ground_truth.append(outflow_ground + 1)

    xgboost_inflow_data = np.asarray(xgboost_inflow_data)
    xgboost_outflow_data = np.asarray(xgboost_outflow_data)
    xgboost_inflow_ground_truth = np.asarray(xgboost_inflow_ground_truth)
    xgboost_outflow_ground_truth = np.asarray(xgboost_outflow_ground_truth)

    return xgboost_inflow_data, xgboost_outflow_data, xgboost_inflow_ground_truth, xgboost_outflow_ground_truth

if __name__ == '__main__':

    # ===============================================================
    # load data
    ts = time.time()

    data, mmn = utils.get_data(dataset, len_global, len_local, neighbor_size, data_choice)

    print('\n Load data elapsed time : %.3f seconds\n' % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    # evaluate model
    ts = time.time()

    xgboost_inflow_data, xgboost_outflow_data, xgboost_inflow_ground_truth, xgboost_outflow_ground_truth = \
        get_data(data['predict_time_test'], neighbor_size, data['index_cut_test'])

    # special, add one model parameter for different external size, due to different dataset
    if dataset == 'bj_taxi':
        error_low = 1e-3
    else:
        error_low = 1e-1

    rmse_score, mape_score = XGBOOST.fit_and_predict(
        xgboost_inflow_data, xgboost_outflow_data,
        xgboost_inflow_ground_truth, xgboost_outflow_ground_truth, args.deep, error_low, mmn._max)

    test_loss = 'rmse (norm): %.6f rmse (real): %.6f, mape: %.6f' \
                % (rmse_score, rmse_score * (mmn._max - mmn._min) / 2., mape_score)
    print(test_loss)

    print("Evaluate elapsed time: %.3f seconds\n" % (time.time() - ts))

    # ===============================================================
    # Record the model result
    current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
    rmse_norm = '{:.2f}'.format(rmse_score * (mmn._max - mmn._min) / 2.)
    mape = '{:.2f}'.format(mape_score * (mmn._max - mmn._min) / 2.)
    header = \
        '{:>10s} {:>15s} {:>5s} {:>10s} {:>10s} {:>20s} {:>8s} {:>8s}'\
            .format(
            'len_local', 'neighbor_size', 'deep', 'start_time', 'end_time', 'time', 'rmse', 'mape')

    hyperparams = \
        '{:>10s} {:>15s} {:>5s} {:>10s} {:>10s} {:>20s} {:>8s} {:>8s}'\
            .format(str(len_local), str(neighbor_size), str(args.deep), data['predict_time_test'][0],
            data['predict_time_test'][-1], current_time, str(rmse_norm), str(mape))

    result_file_name = '{}_{}'.format(dataset, exp_name)
    utils.write_record(result_file_name, header, hyperparams)
