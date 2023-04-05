# Copyright 2023 Bytedance Ltd. and/or its affiliates 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import argparse
import os, time, datetime, json
import numpy as np
import torch


def init():

    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--model', type=str,default='TCNN', help="file name of the model, in src/models; available model: [ValueNetwork]")
        # path
        parser.add_argument('--data_root_path', type=str,default='./db_benchmark_datasets', help="root of all datasets")
        parser.add_argument('--dataset', type=str,default='join-order-benchmark', help="available datasets: [join-order-benchmark, TPCH]")
        parser.add_argument('--save_path', type=str,default="", help="path to save weights")
        parser.add_argument('--meta_save_path', type=str,default="", help="path to save executed datasets")
        parser.add_argument('--dbms', type=str,default="postgresql", help="DBMS: [postgresql, mysql, ]")

        parser.add_argument('--lr', type=float,default=0.001, help="the learning rate")
        parser.add_argument('--epochs', type=int,default=300)
        parser.add_argument('--batch_size', type=int,default=128)
        parser.add_argument('--tolerance', type=int,default=10, help="early stopping tolerance step")
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--npseed', type=int, default=42, help='random seed')
        parser.add_argument('--candidate_list', type=int, default=30, help='the length of candidate list')
        parser.add_argument('--sample_num', type=int, default=1000, help='the length of candidate list')
        parser.add_argument('--pairwise', type=bool, default=False, help='whether to use pairwise. Need to disable sampling')
        parser.add_argument('--split_ratio', type=float, default=0.1, help='ratio of validation in train')
        parser.add_argument('--weighted', action='store_true')
        parser.set_defaults(weighted = False)
        parser.add_argument('--NUM_HINT_SET', type=int, default=49, help='ratio of validation in train')
        parser.add_argument('--app', type=str, default="Bao", help='[Bao, pair, list]')
        parser.add_argument('--shuffle_num', type=int, default=1, help='Uni-one model shuffle JOB')
        parser.add_argument('--train', type=int, default=1, help='Is model training')

        parser.add_argument('--slow', type=int, default=0, help='evaluate')
        parser.add_argument('--grain', type=str, default="adhoc", help='{query, template}, split test set by')
        parser.add_argument('--execute_sql', type=int, default=0, help='execute sql')
        parser.add_argument('--sampling', type=int, default=-1, help='using cross-query sampling strategy')
        parser.add_argument('--pre_execute_all', type=int, default=0, help='using cross-query sampling strategy')

        parser.add_argument('--parameter', type=str, default="", help='evaluat')
        parser.add_argument('--expname', type=str, default="main", help='save file name suffix')
        parser.add_argument('--exp', type=str, default="single", help='experimental scenario. [single, cross, one]')
        parser.add_argument('--src', type=str, default="JOB", help='cross workload transfer training dataset')
        parser.add_argument('--tgt', type=str, default="TPCH", help='cross workload transfer test dataset')


        return parser.parse_args()
    args = parse_args()

    args.meta_save_path = os.path.join('record', args.dbms, args.dataset, 'execute_sql')

    return args


args = init()

import random
def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'test set random seed and torch seed is set to {seed}')
def numpy_same_seeds(seed):
    np.random.seed(seed)
    print(f'validation set numpy random seed to {seed}')

same_seeds(args.seed)
numpy_same_seeds(args.npseed)

import os, sys, io
def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def make_print_to_file(path, execute=0):
    '''
    example:
    use make_print_to_file() , and the all the information of funtion print , will be write in to a log file
    :param path: the path to save print information
    :return: none
    '''
    mkdir(path)
    class Logger(object):
        def __init__(self, filename="Default.log", path=path):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.terminal.flush()
            self.log.flush()
        def flush(self):
            pass
    if args.train:
        sys.stdout = Logger(f'model_{args.expname}.log', path=path)
    else:
        sys.stdout = Logger(f'inference_{args.expname}.log', path=path)
    print(f'all print will write to path {path}')
def nowtime():
    return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable