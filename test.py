# -*- coding: utf-8 -*-
import os
import random
import argparse
import subprocess
import yaml
import re

from keras.layers import  Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.layers import (
Input, Activation, merge, Dense, Reshape, Embedding, Flatten, Dropout, Lambda, LSTM, ConvLSTM2D
)
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
import os
import math
import csv
import model.ASTGCN

def write_result():
    cache_result = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/result'
    train_loss = 'train'
    test_loss = 'test'

    # model_name = 'GLST_Net'
    hyperparams_setting = \
        '{:<25s} {}\n{:<25s} {}\n{:<25s} {}\n{:<25s} {}\n{:<25s} {}\n{:<25s} {}' \
            .format(
            'global length:', 1,
            'local length:', 2,
            'neighbor size:', 3,
            'res unit num:', 4,
            'cnn unit num:', 5,
            'isBN:', 'True' if False else 'False')

    filename = os.path.join(cache_result, 'GLST_Net')
    with open(filename, mode='a') as f:
        f.write(hyperparams_setting + '\n')
        f.write('-' * 30 + '\n')
        f.write(train_loss + '\n')
        f.write(test_loss + '\n')
        f.write('\n')

def arima():


    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    test_data = list([random.randint(1, 20) for i in range(49)])

    # 时序图
    data = pd.Series(test_data)
    data.plot()
    plt.show()

    # 自相关
    from statsmodels.graphics.tsaplots import plot_acf
    #plot_acf(data).show()

    # 平稳性检测
    from statsmodels.tsa.stattools import adfuller as ADF
    print('original ADF result is', ADF(data))

    D_data = data.diff(3).dropna()
    #print(D_data)
    D_data.plot()
    plt.show()
    print('Diffenciate ADF result is', ADF(D_data))

    from statsmodels.tsa.arima_model import ARIMA

    # 定阶
    pmax = int(len(D_data) / 10)
    qmax = int(len(D_data) / 10)
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    print(bic_matrix)
    # 展平后找出最小的位置
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().idxmin()
    print('BIC minimum p and q is', p, q)
    model = ARIMA(data, (p, 1, q)).fit()
    model.summary2()
    model.forecast(5)

def get_data(len_global, len_local, neighbor_size):
    filename = 'data_global_{}_local_{}_neighbor_{}.h5' \
        .format(len_global, len_local, neighbor_size)
    filepath = os.path.join(cache_data_path, filename)
    return os.path.isfile(filepath)

def parse_arg():
    parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
    parser.add_argument('--name', type=str, default='taxi', help='taxi or bike')
    parser.add_argument('--cnn', type=int, default=128,
                        help='dimension of local conv output')
    args = parser.parse_args()
    print(args.cnn)

def run(script_name, log_name, params_name, params_dict):

    def run_command(params, command_set):
        if len(params) == len(params_name):
            command_string = 'nohup python {}'.format(script_name)
            for i in range(len(params)):
                command_string = command_string + ' --{} {}'.format(params_name[i], params[i])
            command_string = command_string + ' >> log/{} 2>&1 &'.format(log_name)
            # print(command_string)
            command_set.append(command_string)
            return
            # print command after run

        current_param_index = len(params)
        current_param_name = params_name[current_param_index]
        for value in params_dict[current_param_name]:
            params.append(value)
            run_command(params, command_set)
            del params[current_param_index]

    return run_command

def shell():
    return ["python temp.py", "python temp.py", "python temp.py"]

def read_yaml():

    cur_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(cur_path, 'params.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)['ha']

    return dict

def spell_string(param_dict):
    left_part, right_part = '{:>', 's}'
    header = ''
    for i, (key, value) in enumerate(param_dict['header'].items()):
        if i == 0:
            header = left_part + str(value) + right_part
        else:
            header = header + ' ' + left_part + str(value) + right_part
        header = header.format(key)

    print(header)

def rename_file():
    path = r'/home/ryj/renyajie/exp/GLST_Net/inter_data/model/'
    file_list = os.listdir(path)

    for file_name in file_list:
        old_name = os.path.join(path, file_name)
        if os.path.isdir(old_name):
            continue
        if old_name.find("lr_0.0001") != -1:
            new_name = old_name.replace("_lr_0.0001", "")
            print("old name", old_name)
            print("new name", new_name)
            os.rename(old_name, new_name)
        # print(new_name)

def science_mode():
    content = "12_5e-05_36"
    # science xe-0x --> xe-x
    pattern_science = re.compile(r'\de-0\d')
    result_science = re.search(pattern_science, content)
    if result_science is not None:
        content = re.sub(r'-0', '-', content)
        print(content)

def get_cheb_polynomials(file_path, num_of_vertices, K):
    adj_mx = model.ASTGCN.get_adjacency_matrix(file_path, num_of_vertices)
    L_tilde = model.ASTGCN.scaled_Laplacian(adj_mx)
    cheb_polynomials = np.array(model.ASTGCN.cheb_polynomial(L_tilde, K))
    return cheb_polynomials

def generate_distance():
    size1, size2 = 32, 32
    region_count = size1 * size2

    pair_distance = dict()
    theta = 5 #2-bike, 5-taxi
    threshold = 0.001 #0.001-bike & taxi
    for row in range(size1):
        for column in range(size2):
            # calculate for every node
            for region in range(region_count):
                i, j = region / size2, region % size2
                # exist or self, skip
                key = (row * size2 + column, region)
                reverse_key = (region, row * size2 + column)
                if key in pair_distance or (i == row and j == column):
                    continue
                cost = math.exp(-(pow((i - row), 2) + pow((j - column), 2)) / (2 * theta * theta))
                distance = 0 if cost < threshold else cost
                pair_distance[key] = distance
                pair_distance[reverse_key] = distance

    with open('/home/ryj/renyajie/exp/GLST_Net/data/taxi_distance.csv', "w", newline='') as f:
        # with open(birth_weight_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to", "cost"])
        for (f, t), cost in pair_distance.items():
            writer.writerow([f, t, cost])

if __name__ == '__main__':
    cheb_polynomials = get_cheb_polynomials('/home/ryj/renyajie/exp/GLST_Net/data/taxi_distance.csv', 32 * 32, 3)
    print(cheb_polynomials.shape)
    #generate_distance()