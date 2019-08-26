# -*- coding=utf-8 -*-
import argparse
import time
from preprocess import utils

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--gpu', type=str, default="1", help='use which gpu 0, 1 or 2')
parser.add_argument('--res', type=int, default=1, help='dimension of local res output')
parser.add_argument('--cnn', type=int, default=2, help='dimension of local conv output')
args = parser.parse_args()

res=args.res
cnn=args.cnn

if __name__ == '__main__':
    # create some error to test the error handle
    time.sleep(3)
    #if args.cnn == 2:
    #    print("this is my fault.")
    #    exit(1)
    print('res {}, cnn {}'.format(args.res, args.cnn))

    current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())

    header = \
        '{:>10s} {:>10s} {:>20s}' \
            .format(
            'res', 'cnn', 'time')

    hyperparams = \
        '{:>10s} {:>10s} {:>20s}' \
            .format(str(res), str(cnn), current_time)

    utils.write_record('temp', header, hyperparams)
