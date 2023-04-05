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
from src.db_util import *
from src.argument import *
import torch


apps = ['Bao', 'pair_wo_weighted', 'list']
grains = ["adhoc", "repeat"]
slows = ["", "/slow_split"]

def extract_speedup(line):
    line = line.split(' ')
    line = float(line[-4].strip(','))
    return line

def analysis_log(log_file_path):
    with open(log_file_path, 'r') as f:
        result = f.read()
    results = result.split('all print')[-10:]

    job_test = []
    tpch_test = []
    tpchs_test = []
    train_times = []
    for i in range(len(results)):
        res = results[i]
        res = res.split('\n')
        train_time = res[-3]
        train_time = train_time.split(' ')[-1]
        train_time = train_time.split('.')[0]
        train_time = datetime.datetime.strptime(train_time, "%H:%M:%S")
        train_times.append(train_time)
        job = extract_speedup(res[-6])
        tpch = extract_speedup(res[-5])
        tpchs = extract_speedup(res[-4])
        job_test.append(job)
        tpch_test.append(tpch)
        tpchs_test.append(tpchs)
    train_times2 = [(i - datetime.datetime(1900, 1,1,0, 0, 0,0)).total_seconds()  for i in train_times]
    print('training time', np.mean(train_times2))
    return np.mean(job_test), np.mean(tpch_test), np.mean(tpchs_test)



for slow in slows:
    for grain in grains:
        print("="*20)
        for app in apps:
            basic_path = os.path.join('record', 'postgresql')
            file_path = os.path.join(basic_path, f'cross/one/TCNN/{slow}', app, f'model_main_{grain}.log') 
            job, tpch, tpchs = analysis_log(file_path)
            print(f'{app}, {grain}, {slow}, test, JOB {job:.3f}, TPCH {tpch:.3f}')

