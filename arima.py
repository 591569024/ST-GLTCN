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
from model import ARIMA
import h5py
import numpy as np
import time
import pickle
import argparse
import template

exp_name = 'arima'

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--dataset', type=str, default='taxi', help='dataset, beijing taxi default')
parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
parser.add_argument('--seq_len', type=int, default=130, help='the number of history data to predict for a region')
parser.add_argument('--region_len', type=int, default=100, help='how many regions to meditate')

args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

len_global = 7  # the length of global time slots
len_local = 4  # the length of local time slots
neighbor_size = 2  # the size of neighbor

seq_len = args.seq_len
region_len = args.region_len

data_choice={}
data_choice['necessary'] = True
data_choice['local_external'] = False
data_choice['local_flow'] = False
data_choice['global_external'] = False
data_choice['global_flow'] = False

dataset = 'bj_taxi' if args.dataset == 'taxi' else 'ny_bike'

def get_data(dataset, predict_time, neighbor_size, seq_len, region_len, index_cut):
    """
    get the arima data
    :param predict_time:
    :param neighbor_size:
    :param seq_len: how many observations contains
    :param region_len: how many regions
    :param index_cut:
    :return:
    """
    slide_length = neighbor_size * 2 + 1
    arima_inflow_data = []
    arima_outflow_data = []
    arima_inflow_ground_truth = []
    arima_outflow_ground_truth = []

    complete_date, complete_data, _, date_index = LoadData.load_bj_taxi_flow() \
        if dataset == 'bj_taxi' else LoadData.load_ny_bike_flow()

    for index, (row_offset, column_offset) in enumerate(index_cut[:region_len]):
        end_index = date_index[predict_time[index]]
        for row in range(slide_length):
            for column in range(slide_length):
                # if length two small, p is abnormal
                arima_inflow_data.append(
                    complete_data[end_index-seq_len:end_index, 0, row+row_offset, column+column_offset])
                arima_outflow_data.append(
                    complete_data[end_index-seq_len:end_index, 1, row+row_offset, column+column_offset])

                arima_inflow_ground_truth.append(
                    complete_data[end_index, 0, row+row_offset, column+column_offset])
                arima_outflow_ground_truth.append(
                    complete_data[end_index, 1, row+row_offset, column+column_offset])


    return arima_inflow_data, arima_inflow_ground_truth, arima_outflow_data, arima_outflow_ground_truth

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

    arima_inflow_data, arima_inflow_ground_truth, arima_outflow_data, arima_outflow_ground_truth = \
        get_data(dataset, data['predict_time_test'], neighbor_size, seq_len, region_len, data['index_cut_test'])

    rmse_score, mape_score = ARIMA.evaluate(
        arima_inflow_data, arima_inflow_ground_truth,
        arima_outflow_data, arima_outflow_ground_truth)

    test_loss = 'rmse (norm): %.6f rmse (real): %.6f' \
                % (rmse_score, rmse_score * (mmn._max - mmn._min) / 2.)
    print(test_loss)

    print("Evaluate elapsed time: %.3f seconds\n" % (time.time() - ts))

    # ===============================================================
    # Record the model result
    current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
    rmse_norm = '{:.2f}'.format(rmse_score * (mmn._max - mmn._min) / 2.)
    mape = '{:.2f}'.format(mape_score * (mmn._max - mmn._min) / 2.)

    header = \
        '{:>10s} {:>10s} {:>15s} {:>10s} {:>10s} {:>20s} {:>8s} {:>8s}'\
            .format(
            'region_len', 'seq_len', 'neighbor_size', 'start_time', 'end_time', 'time', 'rmse', 'mape')

    hyperparams = \
        '{:>10s} {:>10s} {:>15s} {:>10s} {:>10s} {:>20s} {:>8s} {:>8s}'\
            .format(str(region_len), str(seq_len), str(neighbor_size), data['predict_time_test'][0],
            data['predict_time_test'][-1], current_time, str(rmse_norm), str(mape))

    result_file_name = '{}_{}'.format(dataset, exp_name)
    utils.write_record(result_file_name, header, hyperparams)

