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


NUM_HINT_SET = 49 # 48 hint sets + 1 PG default


def file_exist(saved_files, sql_id):
    
    saved_ids =[]
    for i in saved_files:
        saved_ids.append(i.split('.')[0])
    sql_id = sql_id.split('.')[0]
    # print(saved_ids, sql_id)
    if sql_id in saved_ids:
        file_sql_hint_path = os.path.join(args.meta_save_path, f'{sql_id}.sql.plan')
        with open(file_sql_hint_path, 'r') as f:
            tmp = json.load(f)
        if(len(tmp) == 49):
            return True
        else:
            return False
    else:
        return False



def main():
    exe_plan_dict = {} # plan -> latency
    hint_latency_path = os.path.join(args.meta_save_path, f'hint_latency.json')
    try:
        print('try to load plan latency dict')
        with open(hint_latency_path, 'r') as f:
            exe_plan_dict = json.load(f)
    except:
        print('load plan latency dict failed')
    save_folder = args.meta_save_path
    saved_files = os.listdir(save_folder)
    for index, (fp, q) in enumerate(queries):
        file_sql_hint_path = os.path.join(args.meta_save_path, f'{fp}.plan')
        file_sql_latency_path = os.path.join(args.meta_save_path, f'{fp}.latency')
        saved_files = os.listdir(save_folder)
        sql_folder = os.path.join(args.data_root_path, args.dataset)
        fp_full = os.path.join(sql_folder, fp)
        if 1:
            plans = []
            latencys = []
            for i in range(NUM_HINT_SET):
                hints = arm_idx_to_hints_v2(i)
                plan = hints_to_plan(q, hints)
                if str(plan) in exe_plan_dict.keys():
                    latency = exe_plan_dict[str(plan)]
                else:
                    latency = run_query_hint(q, hints) # default PG
                    exe_plan_dict[str(plan)] = latency
                plans.append(plan)
                latencys.append(latency)
                print(f'{index+1}/{len(queries)} query, {fp} the {i} hint set, execute latency {latency:.5f}s, current {len(exe_plan_dict)} unique plans')
            with open(file_sql_hint_path, 'w') as f:
                json.dump(plans, f)
            with open(file_sql_latency_path, 'w') as f:
                json.dump(latencys, f)
            # record_query.append(fp)
            with open(hint_latency_path, 'w') as f:
                json.dump(exe_plan_dict, f)

        print('execute finished, the plans and exe latency saved')
    with open(hint_latency_path, 'w') as f:
        json.dump(exe_plan_dict, f)
    print('plan latency dict saved')

    

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    make_print_to_file(args.meta_save_path, 1)
    print('='*20 + 'start time: ' + nowtime() + '='*20)
    tmp = read_dataset_queries(args)
    queries = []
    if args.dataset == 'TPCH':
        for i in tmp:
            if i[0][:2] != '15':
                # 15 create view wrong
                queries.append(i)
    else:
        queries = tmp

    main()
    t2 = datetime.datetime.now()
    print(f'execute {len(queries)} queries finished')
    print(''*10 + f'execution cost {t2-t1}')
    print('='*20 + 'end time: ' + nowtime() + '='*20)