# -*- coding: utf-8 -*-
"""
define some useful function
"""
import sys
sys.path.append('/home/ryj/renyajie/exp/GLST_Net/preprocess')

import numpy as np
import h5py
import time
import os
import random
import pickle
import LoadData
import yaml

total_size = 500
test_proportion = 0.1

def get_data_path():
    return r"/home/ryj/renyajie/exp/GLST_Net/data"

def get_pkl_path():
    return r"/home/ryj/renyajie/exp/GLST_Net/preprocess"

def remove_incomplete_day(date, data, unit_len=48):
    """
    将不完整的数据剔除，每一天的数据是1-48
    args:
        date: 原始日期
        data：原始日期对应的数据
        unit_len：每一天分为多少个单元
    ret:
        完整的时间和数据
    """

    def get_unit(date):
        return int(date[-2:])

    # get the index of complete data
    index = []
    cur_index = 0

    max_len = len(date)
    while cur_index < max_len:
        cur_unit = get_unit(date[cur_index])

        if cur_unit == 1 and cur_index + unit_len - 1 < max_len \
                and get_unit(date[cur_index + unit_len - 1]) == unit_len:
            index.extend(list(range(cur_index, cur_index + unit_len)))
            cur_index += unit_len
        else:
            cur_index += 1

    # 将不完整的数据过滤
    complete_data = np.array(list(map(lambda x: data[x], index)))
    complete_date = np.array(list(map(lambda x: date[x], index)))

    return complete_date, complete_data

def pick_region(recent_local_data, daily_local_data, week_local_data, ground_truth_data,
                current_flow_data, region_num, neighbor_size, width=32, height=32):
    """
    random pick m regions without overlap in order to reduce the number of local flow and truth
    :param local_flow_data: ~
    :param ground_truth_data: ~
    :param current_flow_data: current map
    :param region_num: how many regions
    :param neighbor_size: ~
    :return: the picked local flow data, ground truth, index_list
    """

    region_slide = neighbor_size * 2 + 1

    # a random start from [0, region_slide) and generate list
    if width == 32:
        row_start = random.randint(0, region_slide - 1)
        column_start = random.randint(0, region_slide - 1)
        row_list = list(range(row_start, width - region_slide, region_slide))
        column_list = list(range(column_start, height - region_slide, region_slide))
    else:
        # if dataset is nyc bike, shrink the step
        row_list = list(range(0, width - region_slide))
        column_list = list(range(0, height - region_slide))

    # generate the all the pairs and pick m from it
    total_pair = []
    for row in row_list:
        for column in column_list:
            total_pair.append((row, column))

    index_list = random.sample(range(0, len(total_pair)), region_num)

    # pick the local flow data
    recent_local_flow = []
    daily_local_flow = []
    week_local_flow = []
    ground_truth = []
    current_flow = []
    index_cut = []
    for index in index_list:
        row, column = total_pair[index]
        index_cut.append((row, column))
        recent_local_flow.append(recent_local_data[:, :, row:row + region_slide, column:column + region_slide])
        daily_local_flow.append(daily_local_data[:, :, row:row + region_slide, column:column + region_slide])
        week_local_flow.append(week_local_data[:, :, row:row + region_slide, column:column + region_slide])
        ground_truth.append(ground_truth_data[:, row:row + region_slide, column:column + region_slide])
        current_flow.append(current_flow_data[:, row:row + region_slide, column:column + region_slide])

    return np.asarray(recent_local_flow), np.asarray(daily_local_flow), np.asarray(week_local_flow)\
        , np.asarray(ground_truth), np.asarray(current_flow), np.asarray(index_cut)

def get_flow_data(date, data, len_global=7, len_local=4, neighbor_size=2, region_num=5,
                  unit_len=48, width=32, height=32):
    """
    get global and local flow data, ground truth and the corresponding date
    :param date: date
    :param data: corresponding data
    :param len_global: the length of time slots for global flow data
    :param len_local: the length of time slots for local flow data
    :param neighbor_size: the size of neighbor, size == (val * 2 + 1)^2
    :param region_num: only consider m region in a map

    :return:
    global flow data, local data, ground truth, prediction date,
    global time slots, local time slots
    """

    # divide one day into 48 units
    max_len = len(data)

    global_flow = []
    stack_local_flow = []
    current_local_flow = []
    ground_truth = []
    index_cut = []
    predict_time = []
    global_timeslots = []
    local_timeslots = []
    for i, one_day in enumerate(date):

        global_start = i - unit_len * (len_global - 1)
        global_end = i
        local_start = i - len_local + 1
        local_end = i

        if global_start >= 0 and global_end < max_len - 1 \
                and local_start >= 0 and local_end < max_len - 1\
                and i < max_len - 1:

            # get data in the exact time
            global_data = data[global_start: global_end + 1: unit_len] # len_global, 2, 32, 32
            local_data = data[local_start: local_end + 1: 1] # len_local, 2, 32, 32
            truth_data = data[local_end + 1] # 2, 32, 32
            current_local_data = data[local_end] # 2, 32, 32

            # random pick m regions without overlap in order to reduce the number of local flow and truth
            pick_local, pick_truth, pick_current, cut_list = \
                pick_region(local_data, truth_data, current_local_data, region_num, neighbor_size,
                            width, height)
            # pick_local ==> m, len_local, 2, a, a
            # pick_truth ==> m, 2, a, a
            # pick_current ==> m, 2, a, a
            # index_list ==> m, 2

            # stack the time slots and channel before storing them
            region_slide = neighbor_size * 2 + 1
            assert global_data.shape == (len_global, 2, width, height)
            assert pick_local.shape == (region_num, len_local, 2, region_slide, region_slide)

            stack_global = np.vstack(global_data)
            stack_local = np.asarray([np.vstack(l) for l in pick_local])

            assert stack_global.shape == (len_global * 2, width, height)
            assert stack_local.shape == (region_num, len_local * 2, region_slide, region_slide)

            global_flow.append(stack_global)
            stack_local_flow.append(stack_local)
            current_local_flow.append(pick_current)
            ground_truth.append(pick_truth)
            index_cut.append(cut_list)
            predict_time.append(date[i + 1])

            # record the time slot for global and local, and encode to ascii
            global_timeslots.append(date[global_start: global_end + 1: unit_len])
            local_timeslots.append(date[local_start: local_end + 1: 1])


    global_flow = np.asarray(global_flow)
    stack_local_flow = np.asarray(stack_local_flow)
    ground_truth = np.asarray(ground_truth)
    current_local_flow = np.asarray(current_local_flow)
    index_cut = np.asarray(index_cut)
    predict_time = np.asarray(predict_time)
    global_timeslots = np.asarray(global_timeslots)
    local_timeslots = np.asarray(local_timeslots)

    # todo remove this, for print
    print('loading flow and timeslots',
          '\n' + '-' * 30,
          '\nglobal_flow:', global_flow.shape,
          '\nstack_local_flow:', stack_local_flow.shape,
          '\nground_truth:', ground_truth.shape,
          '\ncurrent_local_flow:', current_local_flow.shape,
          '\nindex_cut:', index_cut.shape,
          '\npredict_time:', predict_time.shape,
          '\nglobal_timeslots:', global_timeslots.shape,
          '\nlocal_timeslots:', local_timeslots.shape,
          '\n' + '-' * 30)

    return global_flow, stack_local_flow, ground_truth, current_local_flow\
        , index_cut, predict_time, global_timeslots, local_timeslots

def get_flow_rdw_data(date, data, len_recent=4, len_daily=4, len_week=4, neighbor_size=2, region_num=5,
                  unit_len=48, width=32, height=32):
    """
    get global and local flow data, ground truth and the corresponding date
    :param date: date
    :param data: corresponding data
    :param len_global: the length of time slots for global flow data
    :param len_local: the length of time slots for local flow data
    :param neighbor_size: the size of neighbor, size == (val * 2 + 1)^2
    :param region_num: only consider m region in a map

    :return:
    recent, daily, week, current data, ground truth, prediction date, local time slots
    """

    # divide one day into 48 units
    max_len = len(data)

    recent_local_flow = []
    daily_local_flow = []
    week_local_flow = []
    current_local_flow = []
    ground_truth = []
    index_cut = []
    predict_time = []
    recent_time = []
    daily_time = []
    week_time = []
    current_time = []
    for i, one_day in enumerate(date):
        recent_local_start = i - len_recent
        recent_local_end = i - 1
        daily_local_start = i - len_daily * unit_len
        daily_local_end = i - 1 * unit_len
        week_local_start = i - len_week * unit_len * 7
        week_local_end = i - 1 * unit_len * 7

        if recent_local_start >= 0 and recent_local_end < max_len - 1 \
                and daily_local_start >= 0 and daily_local_end < max_len - 1 \
                and week_local_start >= 0 and week_local_end < max_len - 1 \
                and i < max_len - 1:

            # get data in the exact time
            recent_local_data = data[recent_local_start: recent_local_end + 1]
            daily_local_data = data[daily_local_start: daily_local_end + 1: unit_len]
            week_local_data = data[week_local_start: week_local_end + 1: unit_len * 7]
            truth_data = data[i + 1]
            current_local_data = data[i]

            # random pick m regions without overlap in order to reduce the number of local flow and truth
            pick_recent_local, pick_daily_local, pick_week_local, \
            pick_truth, pick_current, cut_list = \
                pick_region(recent_local_data, daily_local_data, week_local_data,
                            truth_data, current_local_data, region_num, neighbor_size,
                            width, height)

            # stack the time slots and channel before storing them
            region_slide = neighbor_size * 2 + 1
            assert pick_recent_local.shape == (region_num, len_recent, 2, region_slide, region_slide)
            assert pick_daily_local.shape == (region_num, len_daily, 2, region_slide, region_slide)
            assert pick_week_local.shape == (region_num, len_week, 2, region_slide, region_slide)
            assert pick_truth.shape == (region_num, 2, region_slide, region_slide)
            assert pick_current.shape == (region_num, 2, region_slide, region_slide)

            pick_recent_local = np.asarray([np.vstack(l) for l in pick_recent_local])
            pick_daily_local = np.asarray([np.vstack(l) for l in pick_daily_local])
            pick_week_local = np.asarray([np.vstack(l) for l in pick_week_local])

            recent_local_flow.append(pick_recent_local)
            daily_local_flow.append(pick_daily_local)
            week_local_flow.append(pick_week_local)
            current_local_flow.append(pick_current)
            ground_truth.append(pick_truth)
            index_cut.append(cut_list)
            predict_time.append(date[i + 1])

            recent_time.append(date[recent_local_start: recent_local_end + 1: 1])
            daily_time.append(date[daily_local_start: daily_local_end + 1: unit_len])
            week_time.append(date[week_local_start: week_local_end + 1: unit_len * 7])
            current_time.append(date[i:i+1:1])


    recent_local_flow = np.asarray(recent_local_flow)
    daily_local_flow = np.asarray(daily_local_flow)
    week_local_flow = np.asarray(week_local_flow)
    ground_truth = np.asarray(ground_truth)
    current_local_flow = np.asarray(current_local_flow)
    index_cut = np.asarray(index_cut)
    predict_time = np.asarray(predict_time)

    recent_time = np.asarray(recent_time)
    daily_time = np.asarray(daily_time)
    week_time = np.asarray(week_time)
    current_time = np.asarray(current_time)

    # todo remove this, for print
    print('loading flow and timeslots',
          '\n' + '-' * 30,
          '\nrecent_local_flow:', recent_local_flow.shape,
          '\ndaily_local_flow:', daily_local_flow.shape,
          '\nweek_local_flow:', week_local_flow.shape,
          '\nground_truth:', ground_truth.shape,
          '\ncurrent_local_flow:', current_local_flow.shape,
          '\nindex_cut:', index_cut.shape,
          '\npredict_time:', predict_time.shape,
          '\nrecent_time:', recent_time.shape,
          '\ndaily_time:', daily_time.shape,
          '\nweek_time:', week_time.shape,
          '\ncurrent_time:', current_time.shape,
          '\n' + '-' * 30)

    return recent_local_flow, daily_local_flow, week_local_flow, ground_truth, current_local_flow\
        , index_cut, predict_time, recent_time, daily_time, week_time, current_time

def duplicate_rdw_data(data_sets, region_num=5):
    """
    Be careful, two many copy will cause memory error
    duplicate data for convenience(local flow data ===> cnn)
    :param data_sets:
    :param region_num: how many regions that a map contains
    :return: a new data_sets
    """

    # show the original data before duplicating
    before_info = ''
    for index, (name, data) in enumerate(data_sets.items()):
        before_info = before_info + "{: <25s}: data shape {: >25s}\n" .format(name, str(data.shape))

    before_info = before_info + '-' * 30
    print(before_info)

    for index, (name, data) in enumerate(data_sets.items()):
        if name == 'recent_local_flow' or name == 'daily_local_flow' or name == 'week_local_flow' or \
                name == 'ground_truth' or name == 'current_local_flow' or name == 'index_cut':
            data_sets[name] = np.vstack(data) # merge the 1th and 2nd dim
        else:
            data_sets[name] = np.repeat(data, region_num, axis=0) # duplicate the data except local flow

    # show the new data after duplicating
    after_info = ''
    for index, (name, data) in enumerate(data_sets.items()):
        after_info = after_info + "{: <25s}: data shape {: >25s}\n".format(name, str(data.shape))

    after_info = after_info + '-' * 30
    print(after_info)

    return data_sets

def duplicate_data(data_sets, region_num=5):
    """
    Be careful, two many copy will cause memory error
    duplicate data for convenience(local flow data ===> cnn)
    :param data_sets:
    :param region_num: how many regions that a map contains
    :return: a new data_sets
    """

    # show the original data before duplicating
    before_info = ''
    for index, (name, data) in enumerate(data_sets.items()):
        before_info = before_info + "{: <25s}: data shape {: >25s}\n" .format(name, str(data.shape))

    before_info = before_info + '-' * 30
    print(before_info)

    for index, (name, data) in enumerate(data_sets.items()):
        if name == 'stack_local_flow' or name == 'ground_truth' \
                or name == 'current_local_flow' or name == 'index_cut':
            data_sets[name] = np.vstack(data) # merge the 1th and 2nd dim
        else:
            data_sets[name] = np.repeat(data, region_num, axis=0) # duplicate the data except local flow

    # show the new data after duplicating
    after_info = ''
    for index, (name, data) in enumerate(data_sets.items()):
        after_info = after_info + "{: <25s}: data shape {: >25s}\n".format(name, str(data.shape))

    after_info = after_info + '-' * 30
    print(after_info)

    return data_sets

def divide_train_and_test(len_test, data_sets):
    """
    divide the data_sets into train data and test data according to the param len_test
    :param len_test: the test length
    :param data_sets: variety of data
    :return: split data
    """

    train_set = []
    test_set = []
    data_name = []

    for name, data in data_sets.items():
        train_set.append(data[:-len_test])
        test_set.append(data[-len_test:])
        data_name.append(name)

    info = ''
    for index, (name, data) in enumerate(data_sets.items()):
        info = info + "{: <25s}: data shape {: >25s}, train set {: >25s}, test set {: >25s}\n"\
            .format(name, str(data.shape), str(train_set[index].shape), str(test_set[index].shape))

    info = info + '-' * 30
    print(info)

    return train_set, test_set, data_name

def encode_time(timeslots):
    """
    change time for ascii from utf-8
    """
    return [time.encode() for time in timeslots]

def decode_time(timeslots):
    """
    change time for ascii from utf-8
    """
    return [time.decode('utf-8') for time in timeslots]

def cache(dataset, train_set, test_set, data_name, len_global, len_local, neighbor_size):
    """
    cache data
    """

    filename = '{}_data_global_{}_local_{}_neighbor_{}.h5'\
        .format(dataset, len_global, len_local, neighbor_size)
    f = h5py.File(os.path.join(cache_data_path, filename), 'w')

    for index, name in enumerate(data_name):
        f['{}_train'.format(name)] = train_set[index]
        f['{}_test'.format(name)] = test_set[index]

    f.close()

def cache_rdw(dataset, train_set, test_set, data_name, len_recent, len_daily, len_week, neighbor_size):
    """
    cache data
    """

    filename = '{}_data_r{}_d{}_w{}_neighbor_{}.h5'\
        .format(dataset, len_recent, len_daily, len_week, neighbor_size)
    f = h5py.File(os.path.join(cache_data_path, filename), 'w')

    for index, name in enumerate(data_name):
        f['{}_train'.format(name)] = train_set[index]
        f['{}_test'.format(name)] = test_set[index]

    f.close()

def read_cache(dataset, len_global, len_local, neighbor_size, data_choice):
    """
    从缓存中读取输入预处理文件
    """
    mmn = pickle.load(open('/home/ryj/renyajie/exp/GLST_Net/preprocess/{}_preprocessing.pkl'.format(dataset), 'rb'))

    filename = '{}_data_global_{}_local_{}_neighbor_{}.h5'.format(dataset, len_global, len_local, neighbor_size)
    f = h5py.File(os.path.join(cache_data_path, filename), 'r')

    result = {}
    # data key name, divide them in order to load necessary data only
    necessary_name = ['index_cut_train', 'ground_truth_train', 'predict_time_train',
                      'index_cut_test', 'ground_truth_test', 'predict_time_test']

    global_flow_name = ['global_flow_train', 'global_timeslots_train',
                        'global_flow_test', 'global_timeslots_test']

    local_flow_name = ['stack_local_flow_train', 'current_local_flow_train', 'local_timeslots_train',
                       'stack_local_flow_test', 'current_local_flow_test', 'local_timeslots_test']

    global_bj_taxi_external_name = ['g_vacation_train', 'g_hour_train', 'g_dayOfWeek_train', 'g_weather_train', 'g_continuous_external_train',
                            'g_vacation_test', 'g_hour_test', 'g_dayOfWeek_test', 'g_weather_test', 'g_continuous_external_test']

    local_bj_taxi_external_name = ['t_vacation_train', 't_hour_train', 't_dayOfWeek_train', 't_weather_train', 't_continuous_external_train',
                           't_vacation_test', 't_hour_test', 't_dayOfWeek_test', 't_weather_test', 't_continuous_external_test']

    global_ny_bike_external_name = ['g_hour_train', 'g_dayOfWeek_train', 'g_hour_test', 'g_dayOfWeek_test']

    local_ny_bike_external_name = ['t_hour_train', 't_dayOfWeek_train', 't_hour_test', 't_dayOfWeek_test']

    if dataset == 'bj_taxi':
        name_dict = {'necessary': necessary_name, 'global_flow': global_flow_name, 'local_flow': local_flow_name,
                     'global_external': global_bj_taxi_external_name,
                     'local_external': local_bj_taxi_external_name}
    else:
        name_dict = {'necessary': necessary_name, 'global_flow': global_flow_name, 'local_flow': local_flow_name,
                     'global_external': global_ny_bike_external_name,
                     'local_external': local_ny_bike_external_name}

    for data_name, value in data_choice.items():
        if value == True:
            for name in name_dict[data_name]:
                result[name] = f[name][()]

    f.close()

    # change the type of time slots to utf-8
    result['predict_time_train'] = np.array(decode_time(result['predict_time_train']))
    result['predict_time_test'] = np.array(decode_time(result['predict_time_test']))

    if data_choice['global_flow']:
        result['global_timeslots_train'] = np.array([decode_time(batch) for batch in result['global_timeslots_train']])
        result['global_timeslots_test'] = np.array([decode_time(batch) for batch in result['global_timeslots_test']])

    if data_choice['local_flow']:
        result['local_timeslots_train'] = np.array([decode_time(batch) for batch in result['local_timeslots_train']])
        result['local_timeslots_test'] = np.array([decode_time(batch) for batch in result['local_timeslots_test']])

    return result, mmn

def read_cache_rdw(dataset, len_recent, len_daily, len_week, neighbor_size, data_choice):
    """
    从缓存中读取输入预处理文件
    """
    mmn = pickle.load(open('/home/ryj/renyajie/exp/GLST_Net/preprocess/{}_preprocessing.pkl'.format(dataset), 'rb'))

    filename = '{}_data_r{}_d{}_w{}_neighbor_{}.h5'.format(dataset, len_recent, len_daily, len_week, neighbor_size)
    f = h5py.File(os.path.join(cache_data_path, filename), 'r')

    result = {}
    # data key name, divide them in order to load necessary data only
    necessary_name = ['index_cut_train', 'ground_truth_train', 'predict_time_train',
                      'index_cut_test', 'ground_truth_test', 'predict_time_test']

    current_flow_name = ['current_local_flow_train', 'current_time_train',
                        'current_local_flow_test', 'current_time_test']
    recent_flow_name = ['recent_local_flow_train', 'recent_time_train',
                         'recent_local_flow_test', 'recent_time_test']
    daily_flow_name = ['daily_local_flow_train', 'daily_time_train',
                         'daily_local_flow_test', 'daily_time_test']
    week_flow_name = ['week_local_flow_train', 'week_time_train',
                         'week_local_flow_test', 'week_time_test']

    current_bj_taxi_external_name = ['current_vacation_train', 'current_hour_train', 'current_dayOfWeek_train',
                                     'current_weather_train', 'current_continuous_external_train',
                                     'current_vacation_test', 'current_hour_test', 'current_dayOfWeek_test',
                                     'current_weather_test', 'current_continuous_external_test']
    recent_bj_taxi_external_name = ['recent_vacation_train', 'recent_hour_train', 'recent_dayOfWeek_train',
                                     'recent_weather_train', 'recent_continuous_external_train',
                                     'recent_vacation_test', 'recent_hour_test', 'recent_dayOfWeek_test',
                                     'recent_weather_test', 'recent_continuous_external_test']
    daily_bj_taxi_external_name = ['daily_vacation_train', 'daily_hour_train', 'daily_dayOfWeek_train',
                                     'daily_weather_train', 'daily_continuous_external_train',
                                     'daily_vacation_test', 'daily_hour_test', 'daily_dayOfWeek_test',
                                     'daily_weather_test', 'daily_continuous_external_test']
    week_bj_taxi_external_name = ['week_vacation_train', 'week_hour_train', 'week_dayOfWeek_train',
                                     'week_weather_train', 'week_continuous_external_train',
                                     'week_vacation_test', 'week_hour_test', 'week_dayOfWeek_test',
                                     'week_weather_test', 'week_continuous_external_test']

    current_ny_bike_external_name = ['current_hour_train', 'current_dayOfWeek_train',
                                     'current_hour_test', 'current_dayOfWeek_test']
    recent_ny_bike_external_name = ['recent_hour_train', 'recent_dayOfWeek_train',
                                     'recent_hour_test', 'recent_dayOfWeek_test']
    daily_ny_bike_external_name = ['daily_hour_train', 'daily_dayOfWeek_train',
                                     'daily_hour_test', 'daily_dayOfWeek_test']
    week_ny_bike_external_name = ['week_hour_train', 'week_dayOfWeek_train',
                                     'week_hour_test', 'week_dayOfWeek_test']

    if dataset == 'bj_taxi':
        name_dict = {'necessary': necessary_name, 'current_flow': current_flow_name,
                     'recent_flow': recent_flow_name, 'daily_flow': daily_flow_name,
                     'week_flow': week_flow_name,
                     'current_external': current_bj_taxi_external_name,
                     'recent_external': recent_bj_taxi_external_name,
                     'daily_external': daily_bj_taxi_external_name,
                     'week_external': week_bj_taxi_external_name}
    else:
        name_dict = {'necessary': necessary_name, 'current_flow': current_flow_name,
                     'recent_flow': recent_flow_name, 'daily_flow': daily_flow_name,
                     'week_flow': week_flow_name,
                     'current_external': current_ny_bike_external_name,
                     'recent_external': recent_ny_bike_external_name,
                     'daily_external': daily_ny_bike_external_name,
                     'week_external': week_ny_bike_external_name}

    for data_name, value in data_choice.items():
        if value == True:
            for name in name_dict[data_name]:
                result[name] = f[name][()]

    f.close()

    # change the type of time slots to utf-8
    result['predict_time_train'] = np.array(decode_time(result['predict_time_train']))
    result['predict_time_test'] = np.array(decode_time(result['predict_time_test']))

    if data_choice['current_flow']:
        result['current_time_train'] = np.array([decode_time(batch) for batch in result['current_time_train']])
        result['current_time_test'] = np.array([decode_time(batch) for batch in result['current_time_test']])

    if data_choice['recent_flow']:
        result['recent_time_train'] = np.array([decode_time(batch) for batch in result['recent_time_train']])
        result['recent_time_test'] = np.array([decode_time(batch) for batch in result['recent_time_test']])

    if data_choice['daily_flow']:
        result['daily_time_train'] = np.array([decode_time(batch) for batch in result['daily_time_train']])
        result['daily_time_test'] = np.array([decode_time(batch) for batch in result['daily_time_test']])

    if data_choice['week_flow']:
        result['week_time_train'] = np.array([decode_time(batch) for batch in result['week_time_train']])
        result['week_time_test'] = np.array([decode_time(batch) for batch in result['week_time_test']])

    return result, mmn

def get_data(dataset, len_global, len_local, neighbor_size, data_choice):

    filename = '{}_data_global_{}_local_{}_neighbor_{}.h5' \
        .format(dataset, len_global, len_local, neighbor_size)
    filepath = os.path.join(cache_data_path, filename)

    if os.path.exists(filepath) == False:
        if dataset == 'bj_taxi':
            train_set, test_set, mmn, data_name = \
                LoadData.load_bj_taxi(proportion_test, len_global, len_local, neighbor_size, region_num)
            cache(dataset, train_set, test_set, data_name, len_global, len_local, neighbor_size)
        else:
            train_set, test_set, mmn, data_name = \
                LoadData.load_ny_bike(proportion_test, len_global, len_local, neighbor_size, region_num)
            cache(dataset, train_set, test_set, data_name, len_global, len_local, neighbor_size)

    return read_cache(dataset, len_global, len_local, neighbor_size, data_choice)

def get_rdw_data(dataset, len_recent, len_daily, len_week, neighbor_size, data_choice):

    filename = '{}_data_r{}_d{}_w{}_neighbor_{}.h5' \
        .format(dataset, len_recent, len_daily, len_week, neighbor_size)
    filepath = os.path.join(cache_data_path, filename)

    if os.path.exists(filepath) == False:
        if dataset == 'bj_taxi':
            train_set, test_set, mmn, data_name = \
                LoadData.load_rdw_bj_taxi(proportion_test, len_recent, len_daily, len_week, neighbor_size, region_num)
            cache_rdw(dataset, train_set, test_set, data_name, len_recent, len_daily, len_week, neighbor_size)
        else:
            train_set, test_set, mmn, data_name = \
                LoadData.load_rdw_ny_bike(proportion_test, len_recent, len_daily, len_week, neighbor_size, region_num)
            cache_rdw(dataset, train_set, test_set, data_name, len_recent, len_daily, len_week, neighbor_size)

    return read_cache_rdw(dataset, len_recent, len_daily, len_week, neighbor_size, data_choice)

def write_record(name, header, hyperparams):
    filename = os.path.join(cache_result, name)
    with open(filename, mode='a') as f:
        if os.path.getsize(filename) == 0:
            f.write(header + '\n')
        f.write(hyperparams + '\n')

# ==============================
proportion_test = 0.1  # use 10% to test
region_num = 5 # how many regions that a map contains

# create inter_data
path_data = 'data'  # the file for caching data
path_model = 'model'  # the file for caching model
cache_name = 'inter_data'  # cache file name
result_name = 'result' # result file name

# create file
if os.path.isdir(cache_name) is False:
    os.mkdir(cache_name)

cache_path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data'
cache_data_path = os.path.join(cache_path, path_data)
cache_model_path = os.path.join(cache_path, path_model)
cache_result = os.path.join(cache_path, result_name)

if os.path.isdir(cache_data_path) is False:
    os.mkdir(cache_data_path)
if os.path.isdir(cache_model_path) is False:
    os.mkdir(cache_model_path)
if os.path.isdir(cache_result) is False:
    os.mkdir(cache_result)


if __name__ == '__main__':
    from LoadData import load
    load(test_proportion)

