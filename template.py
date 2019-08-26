# -*- coding: utf-8 -*-
"""
main script
"""

# add into system path
import sys
import os

sys.path.append('..')

from preprocess import utils
import importlib
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import losses
import h5py
import numpy as np
import time
import pickle
import argparse
import yaml
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


def build_model(model, learning_rate):
    # 1e-4
    # 1e-5  cnn 6-1-6  mape-5.86
    # 1e-6  cnn-6-1-6  mape-6.49
    K.set_epsilon(1e-7)

    def rmse(y_true, y_pred):
        return losses.mean_squared_error(y_true, y_pred) ** 0.5

    def mape(y_true, y_pred):
        return losses.mae(y_true, y_pred)

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam, metrics=[rmse, mape])
    # model.summary()

    return model

def eval_together(y_true, pred_y, threshold):
    # only evaluate those value larger than threshold
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return -1
    mape = np.mean(np.abs(y_true[mask] - pred_y[mask]) / y_true[mask])
    rmse = np.sqrt(np.mean(np.square(y_true[mask] - pred_y[mask])))

    return rmse, mape

def run(param_dict):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # build model

    ts = time.time()

    module = importlib.import_module('model.{}'.format(param_dict['model'][0]))
    func = getattr(module, param_dict['model'][1])
    model = build_model(func(param_dict['model'][2]), param_dict['lr'])

    # create file name
    hyperparams_name = \
        '{}_baseline_{}'.format(param_dict['dataset'], param_dict['exp_name']) \
            if param_dict['is_baseline'] else '{}_{}'.format(param_dict['dataset'], param_dict['exp_name'])

    for key, value in param_dict['hyper_params'].items():
        if key == 'lr': continue
        hyperparams_name = hyperparams_name + '_{}_{}'.format(str(key), str(value))

    model_filename = os.path.join(utils.cache_model_path, '{}.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(model_filename, monitor='val_rmse', verbose=0,
                                       save_best_only=True, mode='min')

    print("\nCompiling model elapsed time : %.3f seconds\n" %
          (time.time() - ts))

    print('=' * 30)

    # ===============================================================
    # train model
    ts = time.time()

    # todo uncomment when nail the learning rate
    # if there is a model before, load weight to accelerate the training process
    if(os.path.exists(model_filename)):
        print('find model weight, prepare to load!')
        model.load_weights(model_filename)
        print("=" * 30)


    model.fit(param_dict['train_data'], param_dict['train_ground'],
              epochs=param_dict['epochs'],
              batch_size=param_dict['batch_size'],
              validation_split=0.1,
              callbacks=[early_stopping, model_checkpoint],
              verbose=2)
    model.save_weights(model_filename, overwrite=True)

    print("\nTraining elapsed time: %.3f seconds\n" % (time.time() - ts))
    print('=' * 30)

    # ===============================================================
    # Evaluate model
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()

    model.load_weights(model_filename)

    score = model.evaluate(param_dict['test_data'], param_dict['test_ground'],
                           batch_size=param_dict['batch_size'] * 2, verbose=0)

    dilated_factor = (param_dict['max'] - param_dict['min']) / 2.
    test_loss = 'Test Loss: %.6f rmse (norm): %.6f rmse (real): %.6f, mape: %.6f' \
                % (score[0], score[1], score[1] * dilated_factor, score[2] * dilated_factor)
    print(test_loss)

    print("Evaluate elapsed time: %.3f seconds\n" % (time.time() - ts))
    # ===============================================================
    # Record the model result
    current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
    rmse_norm = '{:.2f}'.format(score[1] * (param_dict['max'] - param_dict['min']) / 2)
    mape = '{:.2f}'.format(score[2] * (param_dict['max'] - param_dict['min']) / 2)

    # general param
    last_five_value = [param_dict['start_time'], param_dict['start_time'],
                       current_time, rmse_norm, mape]

    left_part, right_part = '{:>', 's}'
    header = ''
    hyper_params = ''
    length = len(param_dict['header'])
    for i, (key, value) in enumerate(param_dict['header'].items()):
        if i == 0:
            header = left_part + str(value) + right_part
            hyper_params = left_part + str(value) + right_part
        else:
            header = header + ' ' + left_part + str(value) + right_part
            hyper_params = hyper_params + ' ' + left_part + str(value) + right_part

        header = header.format(key)
        if i < length - 5:
            hyper_params = hyper_params.format(str(param_dict['hyper_params'][key]))
        else:
            hyper_params = hyper_params.format(last_five_value[5 + i - length])

    result_file_name = '{}_{}'.format(param_dict['dataset'], param_dict['exp_name'])
    utils.write_record(result_file_name, header, hyper_params)

if __name__ == '__main__':
    pass
