# -*- coding:utf-8 -*-
import argparse
import subprocess
import os
import yaml
import re

parser = argparse.ArgumentParser(description='main run script')
parser.add_argument('--type', type=int, default='2', help='1: params_name; 2: param_value; 3: the number of params')
parser.add_argument('--exp', type=str, default='self_tcn_nog_rdw', help='the experiment name')
parser.add_argument('--data', type=str, default="taxi", help="which data set do you want to run")
args = parser.parse_args()

def read_yaml(exp_name):

    cur_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(cur_path, 'params.yaml')

    with open(path, 'r', encoding='utf-8') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)[exp_name]

    params_name = []
    for key in dict.keys():
        params_name.append(key)

    return params_name, dict

def get_finish_combination(dataset):
    path = r"/home/ryj/renyajie/exp/GLST_Net/inter_data/result/" + dataset + '_' + args.exp
    if os.path.exists(path) == False:
        return set()

    pattern_faction = re.compile(r'\d\.\d+')
    pattern_bool = re.compile('(True|False)')
    pattern_science = re.compile(r'\de-0\d')
    combinations=set()

    with open(path, 'r') as f:
        for i, content in enumerate(f.readlines()):
            if i == 0:
                continue
            content = '_'.join(re.split('\s+', content.strip())[:params_number])
            # fraction --> xe-x
            result_faction = re.search(pattern_faction, content)
            if result_faction is not None:
                science_model = re.sub(r'\.0+', '', "%e" % float(result_faction.group()))
                science_model = re.sub(r'-\d', '-', science_model)
                content = re.sub(pattern_faction, science_model, content)


            result_science = re.search(pattern_science, content)
            if result_science is not None:
                content = re.sub(r'-0', '-', content)

            # True | False --> 1 | 0
            result_bool = re.search(pattern_bool, content)
            if result_bool is not None:
                if result_bool.group() == 'True':
                    content = re.sub(pattern_bool, "1", content)
                else:
                    content = re.sub(pattern_bool, "0", content)

            combinations.add(content)

    return combinations

def run(params_name, params_dict, combinations):

    def run_command(params, command_set, dataset_command):
        if len(params) == len(params_name):
            command_value = ''
            for i in range(len(params)):
                if i == 0:
                    command_value = str(params[i])
                else:
                    command_value = command_value + "_" + str(params[i])

            if command_value not in combinations:
                command_set.append(dataset_command + "_" + command_value)
            return

        current_param_index = len(params)
        current_param_name = params_name[current_param_index]
        for value in params_dict[current_param_name]:
            params.append(value)
            run_command(params, command_set, dataset_command)
            del params[current_param_index]

    return run_command


if __name__ == '__main__':
    params_name, dict = read_yaml(args.exp)
    params_number = len(params_name)

    # get the combination in the result file
    dataset_dict = {}
    if args.data == 'both':
        dataset_dict = {'bj_taxi': 'taxi', 'ny_bike': 'bike'}
    elif args.data == 'taxi':
        dataset_dict = {'bj_taxi': 'taxi'}
    else:
        dataset_dict = {'ny_bike': 'bike'}

    all_params_value = []
    for key, dataset_command in dataset_dict.items():
        combinations = get_finish_combination(key)
        params_value = []
        run(params_name, dict, combinations)([], params_value, dataset_command)
        all_params_value.extend(params_value)

    params_value = params_name.insert(0, 'dataset')

    if args.type == 1:
        print(params_name)
    elif args.type == 2:
        print(all_params_value)
    elif args.type == 3:
        print(len(params_name))
    else:
        print(len(all_params_value))