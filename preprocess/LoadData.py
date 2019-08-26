# -*- coding: utf-8 -*-
import sys

import os
import h5py
import numpy as np
from preprocess import utils
from copy import copy
import pickle
from datetime import datetime
import math

class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

def MinMaxScale(data):
    """
    transform data in the range [0, 1]
    """
    min = data.min()
    max = data.max()

    # deal with the special situation, min equals max
    if min == max:
        if min > 1:
            count = 0
            while(min > 1):
                min = min / 10
                count = count + 1
            data = data / pow(10, count)

        return data

    return (data - min) / (max - min)

def load_bj_taxi_external(timeslots, description):
    """
    load external data including the continuous data like temperature and wind speed
    and categorical data like theDayOfWeek, Holiday, Weather, timeOfDay
    :param timeslots: the predicted time - list
    :return: concatenation [Econ;Ecat]
    """
    vacation_list = []
    hour_list = []
    dayOfWeek_list = []
    weather_list = []
    continuous_external_list = []

    for time in timeslots:
        weather, temperature, windSpeed = load_bj_taxi_meteorology(time)
        vacation, hour, dayOfWeek = load_other_bj_taxi_external(time)

        continuous_external = np.hstack([temperature[:, None], windSpeed[:, None]])

        vacation_list.append(vacation)
        hour_list.append(hour)
        dayOfWeek_list.append(dayOfWeek)
        weather_list.append(weather)
        continuous_external_list.append(continuous_external)

    vacation_list = np.asarray(vacation_list)
    hour_list = np.asarray(hour_list)
    dayOfWeek_list = np.asarray(dayOfWeek_list)
    weather_list = np.asarray(weather_list)
    continuous_external_list = np.asarray(continuous_external_list)

    # todo remove this, for print
    print('calculating {}'.format(description),
          '\n' + '-' * 30,
          '\nvacation:', vacation_list.shape,
          '\nhour:', hour_list.shape,
          '\ndayOfWeek:', dayOfWeek_list.shape,
          '\nweather:', weather_list.shape,
          '\ncontinuous_external:', continuous_external_list.shape,
          '\n' + '-' * 30)

    return vacation_list, hour_list, dayOfWeek_list, weather_list, continuous_external_list

def load_ny_bike_external(timeslots, description):
    """
    load external data including the continuous data like temperature and wind speed
    and categorical data like theDayOfWeek, Holiday, Weather, timeOfDay
    :param timeslots: the predicted time - list
    :return: concatenation [Econ;Ecat]
    """
    hour_list = []
    dayOfWeek_list = []

    for time in timeslots:
        hour, dayOfWeek = load_other_ny_bike_external(time)

        hour_list.append(hour)
        dayOfWeek_list.append(dayOfWeek)

    hour_list = np.asarray(hour_list)
    dayOfWeek_list = np.asarray(dayOfWeek_list)

    # todo remove this, for print
    print('calculating {}'.format(description),
          '\n' + '-' * 30,
          '\nhour:', hour_list.shape,
          '\ndayOfWeek:', dayOfWeek_list.shape,
          '\n' + '-' * 30)

    return hour_list, dayOfWeek_list

def load_bj_taxi_meteorology(timeslots=None):
    """
    args:
        timeslots: a list of date
    :return wind speed; temperature; weather
    """
    path = os.path.join(utils.get_data_path(), 'TaxiBJ', 'BJ_Meteorology.h5')

    with h5py.File(path, 'r+') as f:
        Temperature = f['Temperature'][()]
        Weather = f['Weather'][()]
        WindSpeed = f['WindSpeed'][()]
        date = f['date'][()]

    # 从numpy byte转化为str
    date = np.array([time.decode('utf-8') for time in date])

    # 创建索引
    index = [np.where(date == time)[0][0] for time in timeslots]

    temperature = []
    windSpeed = []
    weather = []

    # the last time slot used as the external data
    for idx in index:
        temperature.append(Temperature[idx - 1])
        weather.append(np.argwhere(Weather[idx - 1] == 1).squeeze()) # just one index for embed
        windSpeed.append(WindSpeed[idx - 1])

    temperature = np.asarray(temperature)
    weather = np.asarray(weather)
    windSpeed = np.asarray(windSpeed)

    # min-max-scale to wind speed and temperature
    temperature = MinMaxScale(temperature)
    windSpeed = MinMaxScale(windSpeed)

    return weather, temperature, windSpeed

def load_other_bj_taxi_external(timeslots):
    """
    ret:
        if 1 represent the day is a holiday, otherwise 0
        the hour of a day, from 0 to 23
        the day is weekday or weekend, from 0 to 6
    """
    path = os.path.join(utils.get_data_path(), 'TaxiBJ', 'BJ_Holiday.txt')

    with open(path, 'r') as f:
        holiday = np.asarray(list(map(str.strip, f.readlines())))

    date = [time[:-2] for time in timeslots]
    hour = [int(time[-2:]) // 2 for time in timeslots]
    dayOfWeek = [datetime.strptime(time, "%Y%m%d").weekday() for time in date]

    # create vacation
    vacation = []
    for day in date:
        tmp = np.where(holiday == day)[0]
        if len(tmp) == 1:
            vacation.append(1)
        else:
            vacation.append(0)

    vacation = np.asarray(vacation)
    hour = np.asarray(hour)
    dayOfWeek = np.asarray(dayOfWeek)

    return vacation, hour, dayOfWeek

def load_other_ny_bike_external(timeslots):
    """
    ret:
        the hour of a day, from 0 to 23
        the day is weekday or weekend, from 0 to 6
    """

    date = [time[:-2] for time in timeslots]
    hour = [int(time[-2:]) for time in timeslots]
    dayOfWeek = [datetime.strptime(time, "%Y%m%d").weekday() for time in date]

    hour = np.asarray(hour)
    dayOfWeek = np.asarray(dayOfWeek)

    return hour, dayOfWeek

def load_bj_taxi_flow():
    """
    ret:
        complete taxi flow data
    """

    path_list = []
    for year in range(16, 17):
        path_list.append(os.path.join(utils.get_data_path(), 'TaxiBJ',
                                      'BJ{}_M32x32_T30_InOut.h5'.format(year)))
    # print(path_list)

    complete_date = []
    complete_data = []
    for path in path_list:
        with h5py.File(path, 'r+') as f:
            date = np.array(f['date'][()])
            data = np.array(f['data'][()])
            # 1. 缺省值处理，将不完整的一天去掉
            date, data = utils.remove_incomplete_day(date, data)
            # 2. 异常值处理，去掉流动量小于0的值
            data = np.where(data > 0, data, 0)
            complete_date.extend(date)
            complete_data.extend(data)

    complete_data = np.asarray(complete_data)
    mmn = MinMaxNormalization()
    mmn.fit(complete_data)
    complete_data = np.asarray([mmn.transform(d) for d in complete_data])
    print(mmn._max, mmn._min)

    # 把这个对象存储在pkl文件里
    path = os.path.join(utils.get_pkl_path(), 'bj_taxi_preprocessing.pkl')
    fpkl = open(path, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    complete_date = np.array([time.decode('utf-8') for time in complete_date])
    complete_data = np.asarray(complete_data)

    # create index：date->index->data
    index = {}
    for i, date in enumerate(complete_date):
        index[date] = i

    return complete_date, complete_data, mmn, index

def load_ny_bike_flow():
    """
    ret:
        complete taxi flow data
    """

    path = os.path.join(utils.get_data_path(), 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5')
    # print(path_list)

    complete_date = []
    complete_data = []
    with h5py.File(path, 'r+') as f:
        date = np.array(f['date'][()])
        data = np.array(f['data'][()])
        # 1. 缺省值处理，将不完整的一天去掉
        date, data = utils.remove_incomplete_day(date, data, unit_len=24)
        # 2. 异常值处理，去掉流动量小于0的值
        data = np.where(data > 0, data, 0)
        complete_date.extend(date)
        complete_data.extend(data)

    complete_data = np.asarray(complete_data)
    print(complete_data.shape)

    mmn = MinMaxNormalization()
    mmn.fit(complete_data)
    complete_data = np.asarray([mmn.transform(d) for d in complete_data])

    # 把这个对象存储在pkl文件里
    path = os.path.join(utils.get_pkl_path(), 'ny_bike_preprocessing.pkl')
    fpkl = open(path, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    complete_date = np.array([time.decode('utf-8') for time in complete_date])
    complete_data = np.asarray(complete_data)

    # create index：date->index->data
    index = {}
    for i, date in enumerate(complete_date):
        index[date] = i

    return complete_date, complete_data, mmn, index

def quick_load_bj_taxi_external(timeslots, description, length):
    """
    load external data quickly, if time slots is the same
    :param timeslots: the demanded time
    :param description: global or local
    :return:
    """
    start_time = str(timeslots[0][0])
    end_time = str(timeslots[-1][0])
    path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/data'

    filename = 'bj_taxi_external_{}_{}_{}_{}.h5'.format(description, length, start_time, end_time)
    f = h5py.File(os.path.join(path, filename), 'a')

    # encode the time
    encode_time = np.asarray([utils.encode_time(batch) for batch in timeslots])

    # if the same, load directly
    if 'timeslots' in f and (encode_time == f['timeslots'][()]).all():

        print('cache load bj taxi {} external data from {} to {}'.format(description, start_time, end_time))
        print('-' * 30)

        vacation = f['vacation'][()]
        dayOfWeek = f['dayOfWeek'][()]
        weather = f['weather'][()]
        continuous_external = f['continuous_external'][()]
        hour = f['hour'][()]

        f.close()
        return vacation, hour, dayOfWeek, weather, continuous_external

    f.close()

    # else calculate then cache
    f = h5py.File(os.path.join(path, filename), 'w')

    print('calculate bj taxi {} external data from {} to {}'.format(description, start_time, end_time))
    if description == 'global':
        vacation, hour, dayOfWeek, weather, continuous_external = \
            load_bj_taxi_external(timeslots, 'bj taxi global external data')
    else:
        vacation, hour, dayOfWeek, weather, continuous_external = \
            load_bj_taxi_external(timeslots, 'bj taxi local external data')

    print('cache bj taxi {} external data from {} to {}'.format(description, start_time, end_time))

    f['timeslots'] = encode_time
    f['vacation'] = vacation
    f['hour'] = hour
    f['dayOfWeek'] = dayOfWeek
    f['weather'] = weather
    f['continuous_external'] = continuous_external

    f.close()
    return vacation, hour, dayOfWeek, weather, continuous_external

def quick_load_ny_bike_external(timeslots, description, length):
    """
    load external data quickly, if time slots is the same
    :param timeslots: the demanded time
    :param description: global or local
    :return:
    """
    start_time = str(timeslots[0][0])
    end_time = str(timeslots[-1][0])
    path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/data'

    filename = 'ny_bike_external_{}_{}_{}_{}.h5'.format(description, length, start_time, end_time)
    f = h5py.File(os.path.join(path, filename), 'a')

    # encode the time
    encode_time = np.asarray([utils.encode_time(batch) for batch in timeslots])

    # if the same, load directly
    if 'timeslots' in f and (encode_time == f['timeslots'][()]).all():

        print('cache load ny bike {} external data from {} to {}'.format(description, start_time, end_time))
        print('-' * 30)

        dayOfWeek = f['dayOfWeek'][()]
        hour = f['hour'][()]

        f.close()
        return hour, dayOfWeek

    f.close()

    # else calculate then cache
    f = h5py.File(os.path.join(path, filename), 'w')

    print('calculate ny bike {} external data from {} to {}'.format(description, start_time, end_time))
    if description == 'global':
        hour, dayOfWeek = load_ny_bike_external(timeslots, 'ny bike global external data')
    else:
        hour, dayOfWeek = load_ny_bike_external(timeslots, 'ny bike local external data')

    print('cache ny bike {} external data from {} to {}'.format(description, start_time, end_time))

    f['timeslots'] = encode_time
    f['hour'] = hour
    f['dayOfWeek'] = dayOfWeek

    f.close()
    return hour, dayOfWeek

def load_bj_taxi(proportion_test, len_global=7, len_local=4, neighbor_size=2, region_num=5):
    """
    load all the data
    args:
        len_global: the time length of global data
        len_local: the time length of local data
        neighbor_size: the local size, size = (val * 2 + 1) * (val * 2 + 1)
        region_num: how many regions that a map contains
    ret:
        train set and test set, including:
        1. global external and flow data
        2. local external and flow data
        3. ground truth
    """
    date, data, mmn, index = load_bj_taxi_flow()

    # get global and local flow data, ground truth and the corresponding date
    global_flow, stack_local_flow, ground_truth, current_local_flow, index_cut, \
    predict_time, global_timeslots, local_timeslots = \
        utils.get_flow_data(date, data, len_global, len_local, neighbor_size, region_num)

    # get global and local external data
    # todo to slow, change to h5py
    g_vacation, g_hour, g_dayOfWeek, g_weather, g_continuous_external = \
        quick_load_bj_taxi_external(global_timeslots, 'global', len_global)

    t_vacation, t_hour, t_dayOfWeek, t_weather, t_continuous_external = \
        quick_load_bj_taxi_external(local_timeslots, 'local', len_local)

    # change encode to ascii for time
    predict_time = np.asarray(utils.encode_time(predict_time))
    global_timeslots = np.asarray([utils.encode_time(batch) for batch in global_timeslots])
    local_timeslots = np.asarray([utils.encode_time(batch) for batch in local_timeslots])

    # build train set and test set according to the param:len_test
    data_dict = {'global_flow': global_flow, 'stack_local_flow': stack_local_flow, 'ground_truth': ground_truth,
                 'current_local_flow': current_local_flow, 'index_cut': index_cut, 'predict_time': predict_time,
                 'global_timeslots': global_timeslots, 'local_timeslots': local_timeslots,
                 'g_vacation': g_vacation, 'g_hour': g_hour, 'g_dayOfWeek': g_dayOfWeek,
                 'g_weather': g_weather, 'g_continuous_external': g_continuous_external, 't_vacation': t_vacation,
                 't_hour': t_hour, 't_dayOfWeek': t_dayOfWeek, 't_weather': t_weather,
                 't_continuous_external': t_continuous_external}

    data_dict = utils.duplicate_data(data_dict, region_num)

    total_length = g_vacation.shape[0]
    len_test = math.ceil(total_length * proportion_test)
    len_train = total_length - len_test
    print('train set length {:d}\ntest set length {:d}\n{}'.format(len_train, len_test, '-' * 30))

    train_set, test_set, data_name = utils.divide_train_and_test(len_test, data_dict)

    print('predict start: {}\npredict end: {}'
          .format(data_dict["predict_time"][0].decode('utf-8'), data_dict["predict_time"][-1].decode('utf-8')))
    print('-' * 30)

    return train_set, test_set, mmn, data_name

def load_rdw_bj_taxi(proportion_test, len_recent=4, len_daily=4, len_week=4, neighbor_size=2, region_num=5):
    """
    load all the data
    args:
        len_global: the time length of global data
        len_local: the time length of local data
        neighbor_size: the local size, size = (val * 2 + 1) * (val * 2 + 1)
        region_num: how many regions that a map contains
    ret:
        train set and test set, including:
        1. global external and flow data
        2. local external and flow data
        3. ground truth
    """
    date, data, mmn, index = load_bj_taxi_flow()

    # get global and local flow data, ground truth and the corresponding date
    recent_local_flow, daily_local_flow, week_local_flow, ground_truth, current_local_flow \
        , index_cut, predict_time, recent_time, daily_time, week_time, current_time = \
        utils.get_flow_rdw_data(date, data, len_recent, len_daily, len_week,
                                neighbor_size, region_num)

    # get recent, daily, week, current external data
    recent_vacation, recent_hour, recent_dayOfWeek, recent_weather, recent_continuous_external = \
        quick_load_bj_taxi_external(recent_time, 'recent', len_recent)
    daily_vacation, daily_hour, daily_dayOfWeek, daily_weather, daily_continuous_external = \
        quick_load_bj_taxi_external(daily_time, 'daily', len_daily)
    week_vacation, week_hour, week_dayOfWeek, week_weather, week_continuous_external = \
        quick_load_bj_taxi_external(week_time, 'week', len_week)
    current_vacation, current_hour, current_dayOfWeek, current_weather, current_continuous_external = \
        quick_load_bj_taxi_external(current_time, 'current', 1)

    # change encode to ascii for time
    predict_time = np.asarray(utils.encode_time(predict_time))
    recent_time = np.asarray([utils.encode_time(batch) for batch in recent_time])
    daily_time = np.asarray([utils.encode_time(batch) for batch in daily_time])
    week_time = np.asarray([utils.encode_time(batch) for batch in week_time])
    current_time = np.asarray([utils.encode_time(batch) for batch in current_time])

    # build train set and test set according to the param:len_test
    data_dict = {'recent_local_flow': recent_local_flow, 'daily_local_flow': daily_local_flow,
                 'week_local_flow': week_local_flow, 'current_local_flow': current_local_flow,
                 'ground_truth': ground_truth, 'index_cut': index_cut, 'predict_time': predict_time,
                 'recent_time': recent_time, 'daily_time': daily_time, 'week_time': week_time,
                 'current_time': current_time,
                 'recent_vacation': recent_vacation, 'recent_hour': recent_hour,
                 'recent_dayOfWeek': recent_dayOfWeek, 'recent_weather': recent_weather,
                 'recent_continuous_external': recent_continuous_external,
                 'daily_vacation': daily_vacation,
                 'daily_hour': daily_hour, 'daily_dayOfWeek': daily_dayOfWeek, 'daily_weather': daily_weather,
                 'daily_continuous_external': daily_continuous_external,
                 'week_vacation': week_vacation, 'week_hour': week_hour, 'week_dayOfWeek': week_dayOfWeek,
                 'week_weather': week_weather, 'week_continuous_external': week_continuous_external,
                 'current_vacation': current_vacation, 'current_hour': current_hour, 'current_dayOfWeek': current_dayOfWeek,
                 'current_weather': current_weather, 'current_continuous_external': current_continuous_external
    }

    data_dict = utils.duplicate_rdw_data(data_dict, region_num)

    total_length = current_time.shape[0]
    len_test = math.ceil(total_length * proportion_test)
    len_train = total_length - len_test
    print('train set length {:d}\ntest set length {:d}\n{}'.format(len_train, len_test, '-' * 30))

    train_set, test_set, data_name = utils.divide_train_and_test(len_test, data_dict)

    print('predict start: {}\npredict end: {}'
          .format(data_dict["predict_time"][0].decode('utf-8'), data_dict["predict_time"][-1].decode('utf-8')))
    print('-' * 30)

    return train_set, test_set, mmn, data_name

def load_rdw_ny_bike(proportion_test, len_recent=4, len_daily=4, len_week=4, neighbor_size=2, region_num=5):
    """
    load all the data
    args:
        len_global: the time length of global data
        len_local: the time length of local data
        neighbor_size: the local size, size = (val * 2 + 1) * (val * 2 + 1)
        region_num: how many regions that a map contains
    ret:
        train set and test set, including:
        1. global external and flow data
        2. local external and flow data
        3. ground truth
    """
    date, data, mmn, index = load_ny_bike_flow()

    # get global and local flow data, ground truth and the corresponding date
    recent_local_flow, daily_local_flow, week_local_flow, ground_truth, current_local_flow \
        , index_cut, predict_time, recent_time, daily_time, week_time, current_time = \
        utils.get_flow_rdw_data(date, data, len_recent, len_daily, len_week,
                                neighbor_size, region_num, unit_len=24, width=16, height=8)

    # get recent, daily, week, current external data
    recent_hour, recent_dayOfWeek = quick_load_ny_bike_external(recent_time, 'recent', len_recent)
    daily_hour, daily_dayOfWeek = quick_load_ny_bike_external(daily_time, 'daily', len_daily)
    week_hour, week_dayOfWeek = quick_load_ny_bike_external(week_time, 'week', len_week)
    current_hour, current_dayOfWeek = quick_load_ny_bike_external(current_time, 'current', 1)

    # change encode to ascii for time
    predict_time = np.asarray(utils.encode_time(predict_time))
    recent_time = np.asarray([utils.encode_time(batch) for batch in recent_time])
    daily_time = np.asarray([utils.encode_time(batch) for batch in daily_time])
    week_time = np.asarray([utils.encode_time(batch) for batch in week_time])
    current_time = np.asarray([utils.encode_time(batch) for batch in current_time])

    # build train set and test set according to the param:len_test
    data_dict = {'recent_local_flow': recent_local_flow, 'daily_local_flow': daily_local_flow,
                 'week_local_flow': week_local_flow, 'current_local_flow': current_local_flow,
                 'ground_truth': ground_truth, 'index_cut': index_cut, 'predict_time': predict_time,
                 'recent_time': recent_time, 'daily_time': daily_time, 'week_time': week_time,
                 'current_time': current_time,
                 'recent_hour': recent_hour, 'recent_dayOfWeek': recent_dayOfWeek,
                 'daily_hour': daily_hour, 'daily_dayOfWeek': daily_dayOfWeek,
                 'week_hour': week_hour, 'week_dayOfWeek': week_dayOfWeek,
                 'current_hour': current_hour, 'current_dayOfWeek': current_dayOfWeek
    }

    data_dict = utils.duplicate_rdw_data(data_dict, region_num)

    total_length = current_time.shape[0]
    len_test = math.ceil(total_length * proportion_test)
    len_train = total_length - len_test
    print('train set length {:d}\ntest set length {:d}\n{}'.format(len_train, len_test, '-' * 30))

    train_set, test_set, data_name = utils.divide_train_and_test(len_test, data_dict)

    print('predict start: {}\npredict end: {}'
          .format(data_dict["predict_time"][0].decode('utf-8'), data_dict["predict_time"][-1].decode('utf-8')))
    print('-' * 30)

    return train_set, test_set, mmn, data_name

def load_ny_bike(proportion_test, len_global=7, len_local=4, neighbor_size=3, region_num=5):
    """
    load all the data
    args:
        len_global: the time length of global data
        len_local: the time length of local data
        neighbor_size: the local size, size = (val * 2 + 1) * (val * 2 + 1)
        region_num: how many regions that a map contains
    ret:
        train set and test set, including:
        1. global external and flow data
        2. local external and flow data
        3. ground truth
    """
    date, data, mmn, index = load_ny_bike_flow()

    # get global and local flow data, ground truth and the corresponding date
    global_flow, stack_local_flow, ground_truth, current_local_flow, index_cut, \
    predict_time, global_timeslots, local_timeslots = \
        utils.get_flow_data(date, data, len_global, len_local, neighbor_size, region_num,
                            unit_len=24, width=16, height=8)

    # get global and local external data
    g_hour, g_dayOfWeek = quick_load_ny_bike_external(global_timeslots, 'global', len_global)

    t_hour, t_dayOfWeek = quick_load_ny_bike_external(local_timeslots, 'local', len_local)

    # change encode to ascii for time
    predict_time = np.asarray(utils.encode_time(predict_time))
    global_timeslots = np.asarray([utils.encode_time(batch) for batch in global_timeslots])
    local_timeslots = np.asarray([utils.encode_time(batch) for batch in local_timeslots])

    # build train set and test set according to the param:len_test
    data_dict = {'global_flow': global_flow, 'stack_local_flow': stack_local_flow, 'ground_truth': ground_truth,
                 'current_local_flow': current_local_flow, 'index_cut': index_cut, 'predict_time': predict_time,
                 'global_timeslots': global_timeslots, 'local_timeslots': local_timeslots,
                 'g_hour': g_hour, 'g_dayOfWeek': g_dayOfWeek, 't_hour': t_hour, 't_dayOfWeek': t_dayOfWeek}

    data_dict = utils.duplicate_data(data_dict, region_num)

    total_length = g_dayOfWeek.shape[0]
    len_test = math.ceil(total_length * proportion_test)
    len_train = total_length - len_test
    print('train set length {:d}\ntest set length {:d}\n{}'.format(len_train, len_test, '-' * 30))

    train_set, test_set, data_name = utils.divide_train_and_test(len_test, data_dict)

    print('predict start: {}\npredict end: {}'
          .format(data_dict["predict_time"][0].decode('utf-8'), data_dict["predict_time"][-1].decode('utf-8')))
    print('-' * 30)

    return train_set, test_set, mmn, data_name

if __name__ == '__main__':
    timeslots = ['2013020901', '2013030121', '2013020445', '2013020745']
    # load_meteorology(timeslots)
    # load_other_external(timeslots)
    # load_external(timeslots)
    # load_ny_bike(0.1)
    load_bj_taxi_flow()