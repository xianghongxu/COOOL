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
import importlib
from json import load
from src.db_util import *
from src.argument import *
from src.featurize import TreeFeaturizer
from src.ltr.listmleloss import listMLE
from src.ltr.wpmle import pairwiseMLE
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import pandas as pd

from sklearn import preprocessing
from sklearn.pipeline import Pipeline


def load_JOB_data():
    """
    load query ID, plan, latency from record/join-order-benchmark/execute_sql
    query_ID is ordered by os.listdir
    retrun:
        query_ID: [113] 1a.sql etc.
        record_plans: [113, 49]
        record_latency: [113, 49]
    """
    # load query ID
    query_ID = []
    sql_folder = os.path.join(args.data_root_path, 'join-order-benchmark')
    sql_files = os.listdir(sql_folder)
    for fp in sql_files:
        if fp.split('.')[-1] == 'sql':
            query_ID.append(fp) # [1a.sql, 1b.sql, ...]
    print(f'Loaded number of JOB queries: {len(query_ID)}')

    # load {plan, latency} by query
    record_plans = [] # [113, 49] plan
    record_latency = [] # [113, 49] latency
    for fp in query_ID:
        file_sql_hint_path = os.path.join('record', args.dbms, 'join-order-benchmark/execute_sql', f'{fp}.plan')
        file_sql_latency_path = os.path.join('record', args.dbms, 'join-order-benchmark/execute_sql', f'{fp}.latency')
        with open(file_sql_hint_path, 'r') as f:
            plan = json.load(f)
            record_plans.append(plan)
        with open(file_sql_latency_path, 'r') as f:
            latency = json.load(f)
            record_latency.append(latency)
    print(np.array(record_plans).shape)
    print(np.array(record_latency).shape)
    print(query_ID[0])
    return query_ID, record_plans, record_latency

def split_JOB():
    """
    split JOB into train, test, validaiton
    """
    query_ID, record_plans, record_latency = load_JOB_data()
    # select one query from the corresponding template
    if args.grain == 'repeat':
        id_dict = {} # {template id: [index in record_plans] }
        for i, qid in enumerate(query_ID):
            qid = qid.split('.')[0]
            qid = qid[:-1]
            if qid in id_dict.keys():
                id_dict[qid].append(i)
            else:
                id_dict[qid] = []
                id_dict[qid].append(i)
        train_indexs = []
        test_indexs = []
        for i in sorted(id_dict.keys()):
            curr= id_dict[i]

            if args.slow <= 0:
                selected = list([random.choice(curr)]) # random select one
            else:
                pg_time = np.array(record_latency)[:, -1]
                pg_time = np.take(pg_time, curr, axis=0)
                pg_time = torch.tensor(pg_time, dtype=torch.float32)
                sorted_, indexes = torch.sort(pg_time, descending=False)
                selected = list([curr[indexes.numpy()[-1]]])

            curr = list(set(curr).difference(set(selected)))
            test_indexs.extend(selected)
            train_indexs.extend(curr)

        train_plan = np.take(record_plans, train_indexs, axis=0)
        train_latency = np.take(record_latency, train_indexs, axis=0)
        test_plan = np.take(record_plans, test_indexs, axis=0)
        test_latency = np.take(record_latency, test_indexs, axis=0)

    elif args.grain == 'adhoc':
        """
        split 33 * 0.2 = 7 templates for test
        """
        id_dict = {} # {template id: [index in record_plans] }
        for i, qid in enumerate(query_ID):
            qid = qid.split('.')[0]
            qid = qid[:-1]
            if qid in id_dict.keys():
                id_dict[qid].append(i)
            else:
                id_dict[qid] = []
                id_dict[qid].append(i)

        train_indexs = []
        test_indexs = []
        keys = sorted(id_dict.keys())
        key_pg = {}
        pg_time = np.array(record_latency)[:, -1] 
        for i in keys:
            key_pg[i] = np.take(pg_time, id_dict[i], axis=0)
        template_time = [] 
        for i in range(1, 34):
            template_time.append(np.mean(key_pg[str(i)]))
        slow_index = np.argsort(template_time)[-7:]
        slow_index = slow_index + 1
        if args.slow <= 0:
            test_templates = random.sample(keys, 7)
        else:
            test_templates = [str(i) for i in slow_index]
        for i in keys:
            if i in test_templates:
                test_indexs.extend(id_dict[i])
            else:
                train_indexs.extend(id_dict[i])
        print('test templates: ', test_templates)
        print('test queries: ', np.take(query_ID, test_indexs, axis=0))
        train_plan = np.take(record_plans, train_indexs, axis=0)
        train_latency = np.take(record_latency, train_indexs, axis=0)
        test_plan = np.take(record_plans, test_indexs, axis=0)
        test_latency = np.take(record_latency, test_indexs, axis=0)

        
    else:
        print("split grain error")
        exit(0)

    train_query = np.take(query_ID, train_indexs)
    test_query = np.take(query_ID, test_indexs)
    split_num = int(args.split_ratio * len(train_plan))
    indices = np.arange(len(train_plan))
    np.random.shuffle(indices)
    validation_plan = np.take(train_plan, indices[:split_num], axis=0)
    validation_latency = np.take(train_latency, indices[:split_num], axis=0)
    validation_query = train_query[:split_num]
    train_plan = np.take(train_plan, indices[split_num:], axis=0)
    train_latency = np.take(train_latency, indices[split_num:], axis=0)
    train_query = train_query[split_num:]


    query_ids = [train_query, validation_query, test_query]
    print('test query: ', test_query)
    
    return train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency



def load_TPCH_data(skew=False):
    """
    load plan, latency from 'record', args.dbms, TPCH/execute_sql
    retrun:
        query_ID: [200] 20 template, 10 query for each template
        record_plans: [20, 10, 49]
        record_latency: [20, 10, 49]
    """
    # load query ID
    ranges = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22] # exclude 2, 19
    record_plans = [] 
    record_latency = []
    if skew == True:
        meta_save_path = os.path.join('record', args.dbms, 'TPCH_Skew/execute_sql')
        print('load TPCH_Skew...')
    else:
        meta_save_path = os.path.join('record', args.dbms, 'TPCH/execute_sql')
        print('load TPCH...')
    
    for i in ranges:
        template_plan = []
        template_latency = []
        for j in range(1, 11):
            file_sql_hint_path = os.path.join(meta_save_path, f'{i}_{j}.sql.plan')
            file_sql_latency_path = os.path.join(meta_save_path, f'{i}_{j}.sql.latency')
            with open(file_sql_hint_path, 'r') as f:
                plan = json.load(f)
            with open(file_sql_latency_path, 'r') as f:
                latency = json.load(f)
            template_plan.append(plan)
            template_latency.append(latency)
        record_plans.append(template_plan)
        record_latency.append(template_latency)
    return ranges, record_plans, record_latency

def convert_TPCH(a_list):
    """
    [template_num, 10, 49] -> [template_num * 10, 49]
    """
    tmp_array = np.array(a_list)
    tmp_array = np.reshape(tmp_array, (-1, 49))
    ans_list = tmp_array.tolist()
    return ans_list


def split_TPCH(skew=False):
    ranges, record_plans, record_latency = load_TPCH_data(skew)
    test_plan = []
    test_latency = []
    train_plan = []
    train_latency = []
    validation_plan = []
    validation_latency = []
    if args.grain == 'repeat':
        for index, i in enumerate(ranges):
            qids = [j for j in range(10)]
            if args.slow <= 0:
                test_index = random.sample(qids, 2) # 2 query ids
            else:
                pg_time = record_latency[index] # [10, 49]
                pg_time = np.array(pg_time)[:, -1] # [10]
                pg_time = torch.tensor(pg_time, dtype=torch.float32)
                sorted_, indexes = torch.sort(pg_time, descending=False)
                test_index = list([indexes.numpy()[-1], indexes.numpy()[-2]]) # slowest 2 pg execution
            train_index = list(set(qids).difference(set(test_index))) # 8 query
            validation_index = random.sample(train_index, 2)
            train_index = list(set(train_index).difference(set(validation_index))) # 6 query

            print(train_index, test_index, validation_index)

            train_plan.extend(np.take(record_plans[index], train_index, axis=0))
            train_latency.extend(np.take(record_latency[index], train_index, axis=0))
            test_plan.extend(np.take(record_plans[index], test_index, axis=0))
            test_latency.extend(np.take(record_latency[index], test_index, axis=0))
            validation_plan.extend(np.take(record_plans[index], validation_index, axis=0))
            validation_latency.extend(np.take(record_latency[index], validation_index, axis=0))


    elif args.grain == 'adhoc':
        if args.slow <= 0:
            test_template = random.sample(ranges, 4) 
        else:
            pg_time = np.array(record_latency)
            pg_time = torch.tensor(pg_time, dtype=torch.float32)
            pg_time = torch.sum(pg_time, dim=1)
            pg_time = pg_time[:,-1]
            sorted_, indexes = torch.sort(pg_time, descending=True)
            test_template = [ranges[indexes[0]], ranges[indexes[1]], ranges[indexes[2]], ranges[indexes[3]]]
            # [17, 20, 3, 9] 600s+, 540s, 94s, 93s.
        print(f'TPCH test template {test_template}')
        for index, i in enumerate(ranges):
            if i in test_template:
                test_plan.append(record_plans[index])
                test_latency.append(record_latency[index])
            else:
                train_plan.append(record_plans[index])
                train_latency.append(record_latency[index])
        split_num = math.ceil(args.split_ratio * len(train_plan))
        indices = np.arange(len(train_plan))
        np.random.shuffle(indices)
        validation_plan = np.take(train_plan, indices[:split_num], axis=0)
        validation_latency = np.take(train_latency, indices[:split_num], axis=0)
        train_plan = np.take(train_plan, indices[split_num:], axis=0)
        train_latency = np.take(train_latency, indices[split_num:], axis=0)

        train_plan = convert_TPCH(train_plan)
        train_latency = convert_TPCH(train_latency)
        test_plan = convert_TPCH(test_plan)
        test_latency = convert_TPCH(test_latency)
        validation_plan = convert_TPCH(validation_plan)
        validation_latency = convert_TPCH(validation_latency)

    return train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency
 

def pre_evaluate_process(plans):
    tmp = [i for item in plans for i in item]
    featurizer = TreeFeaturizer()
    featurizer.fit(tmp)
    x = [featurizer.transform(i) for i in plans]
    x = np.array(x)
    vectorized_plan = [i for item in x for i in item]
    return vectorized_plan

def obtain_speedup(hint_latency_table, indices):
    """
    hint_latency_table \in [n x 49] latency, indecies indicates the selected arm
    retrun speed up compared with expert optimizer (e.g., PG)
    """
    model_run_train = []
    pg_run_train = []
    for index, armid in enumerate(indices):
        model_time = hint_latency_table[index][armid]
        pg_time = hint_latency_table[index][-1]
        model_run_train.append(model_time)
        pg_run_train.append(pg_time)
    return np.sum(pg_run_train) / np.sum(model_run_train)

def individual_query_performance(hint_latency_table, indices):
    ans = []
    for index, armid in enumerate(indices):
        model_time = hint_latency_table[index][armid]
        pg_time = hint_latency_table[index][-1]
        opt_time = min(hint_latency_table[index])
        ans.append((model_time, pg_time, opt_time))
    return ans

def show_per_query_speedup(ans_per_query):
    for index, i in enumerate(ans_per_query):
        print(f'{index+1:02d}, opt runtime {i[2]:.3f}s, model runtime {i[0]:.3f}s, pg runtime {i[1]:.3f}s, speedup {i[1]/i[0]:.3f}x')


def inference_and_analysis(train_plan, train_latency, test_plans, test_latencies, validation_plan, validation_latency):
    model = trans_to_cuda(config.Model(9)) # in_channel = 9, model define
    saved_model_path = f'{args.save_path}/{args.expname}'
    saved_models = os.listdir(saved_model_path) # [xx.pt, xx.pt, ...]
    saved_models.sort()
    print(saved_models)
    test_speed_record = []
    # load 10 models and execute the inference stage on the test dataset
    for saved_model in saved_models:
        print("inference model and show individual query perf")
        model_path = os.path.join(saved_model_path, saved_model)

        model.load_state_dict(torch.load(model_path)) # model load
        model.eval()
        for i in range(len(test_plans)):
            test_plan = test_plans[i]
            test_latency = test_latencies[i]
            test_plan_vec = pre_evaluate_process(test_plan)
            if args.app == 'pair' or args.app == 'list':
                descending=True
            else:
                descending=False
            scores = model(test_plan_vec)
            scores = torch.reshape(scores, (-1, args.NUM_HINT_SET))  # [num_sql, 49]
            sorted_, indices = torch.sort(scores, dim=-1, descending=descending)
            indices = torch.reshape(indices, (-1, args.NUM_HINT_SET))
            index = indices[:,0].cpu().numpy() # the optimal plan
            test_speedup = obtain_speedup(test_latency, index)
            print(f'total query execution latency speedup {test_speedup} x')
            test_speed_record.append(test_speedup)
            test_query_model_pg = individual_query_performance(test_latency, index)
            show_per_query_speedup(test_query_model_pg)



def main_single():
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    train_plan_job, train_latency_job, test_plan_job, test_latency_job, validation_plan_job, validation_latency_job = split_JOB()
    print(f'load JOB succ')
    train_plan_tpch, train_latency_tpch, test_plan_tpch, test_latency_tpch, validation_plan_tpch, validation_latency_tpch = split_TPCH(skew=False)
    print(f'load TPCH succ')
    if args.dataset=='join-order-benchmark':
        train_plan = train_plan_job
        train_latency = train_latency_job
        test_plan = test_plan_job
        test_latency = test_latency_job
        validation_plan = validation_plan_job
        validation_latency = validation_latency_job
    elif args.dataset=='TPCH':
        train_plan = train_plan_tpch
        train_latency = train_latency_tpch
        test_plan = test_plan_tpch
        test_latency = test_latency_tpch
        validation_plan = validation_plan_tpch
        validation_latency = validation_latency_tpch
    else:
        print("Wrong dataset")
    test_plan = [test_plan]
    test_latency = [test_latency]
    print('train, test, validation',np.array(train_plan).shape, np.array(test_plan).shape, np.array(validation_plan).shape)
    #================ inference
    inference_and_analysis(train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency)


def main_cross():
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # [#query, #hints]
    
    train_plan_job, train_latency_job, test_plan_job, test_latency_job, validation_plan_job, validation_latency_job = split_JOB()
    print(f'load JOB succ')
    train_plan_tpch, train_latency_tpch, test_plan_tpch, test_latency_tpch, validation_plan_tpch, validation_latency_tpch = split_TPCH(skew=False)
    print(f'load TPCH succ')

    if args.src=='join-order-benchmark' or args.src=='JOB':
        train_plan = train_plan_job
        train_latency = train_latency_job
        validation_plan = validation_plan_job
        validation_latency = validation_latency_job
    elif args.src=='TPCH':
        train_plan = train_plan_tpch
        train_latency = train_latency_tpch
        validation_plan = validation_plan_tpch
        validation_latency = validation_latency_tpch
    
    
    if args.tgt=='join-order-benchmark' or args.tgt=='JOB':
        test_plan = test_plan_job
        test_latency = test_latency_job
    elif args.tgt=='TPCH':
        test_plan = test_plan_tpch
        test_latency = test_latency_tpch
    test_plan = [test_plan]
    test_latency = [test_latency]
    print('train, test, validation',np.array(train_plan).shape, np.array(test_plan).shape, np.array(validation_plan).shape)

    inference_and_analysis(train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency)

def main_one():
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # [#query, #hints]
    train_plan_job, train_latency_job, test_plan_job, test_latency_job, validation_plan_job, validation_latency_job = split_JOB()
    print(f'load JOB succ')
    train_plan_tpch, train_latency_tpch, test_plan_tpch, test_latency_tpch, validation_plan_tpch, validation_latency_tpch = split_TPCH(skew=False)
    print(f'load TPCH succ')

    train_plan = np.concatenate((train_plan_job, train_plan_tpch), axis=0)
    train_latency = np.concatenate((train_latency_job, train_latency_tpch), axis=0)
    validation_plan = np.concatenate((validation_plan_job , validation_plan_tpch), axis=0)
    validation_latency = np.concatenate((validation_latency_job , validation_latency_tpch), axis=0)
    test_plan = [test_plan_job, test_plan_tpch]
    test_latency = [test_latency_job, test_latency_tpch]
    # test_plan = np.concatenate((test_plan_job, test_plan_tpch), axis=0)
    # test_latency = np.concatenate((test_latency_job, test_latency_tpch), axis=0)
    inference_and_analysis(train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency)






if __name__ == '__main__':
    args.train = 0 # inference to obtain paper reported results.

    t1_program = datetime.datetime.now()
    save_model_time = nowtime()
    args.expname = args.expname + '_' + args.grain
    args.meta_save_path = os.path.join('record', args.dbms, args.dataset, 'execute_sql')
    if args.exp == 'single':
        args.save_path = os.path.join('record', args.dbms, args.dataset, args.model)
    elif args.exp == 'cross':
        args.save_path = os.path.join('record', args.dbms, f'cross/{args.src+"_"+args.tgt}', args.model)
    else:
        args.save_path = os.path.join('record', args.dbms, f'cross/one', args.model)
    # save log path
    
    if args.slow > 0:
        args.save_path = os.path.join(args.save_path, f'slow_split')
    # three approaches save principle
    if args.app == 'list':
        if args.sampling == 0:
            args.save_path = os.path.join(args.save_path, "wo_sampling")
        else:
            args.save_path = os.path.join(args.save_path, args.app)
    elif args.app == 'pair':
        args.epochs = 50
        # args.lr /= 10
        if args.weighted == False:
            args.save_path = os.path.join(args.save_path, "pair_wo_weighted")
        else: 
            args.save_path = os.path.join(args.save_path, args.app)
    else:
        args.save_path = os.path.join(args.save_path, args.app)
    
    # load model
    module = importlib.import_module(f'src.models.{args.model}')
    config = module.config
    for k, v in vars(args).items():
        config[k] = v
    args = config

    make_print_to_file(args.save_path)
    print('='*50)
    print(args)
    if args.exp == 'single':
        main_single() # single instance experiments
    elif args.exp == 'cross':
        main_cross() # cross workload experiments
    else:
        main_one() # unified model experiments

    t2_program = datetime.datetime.now()
    print(''*10 + f'execution cost {t2_program-t1_program}')
    print('='*50)
