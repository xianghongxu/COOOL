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






def hints_2_plan(view, sql, close, hints):
    '''
    input: a string SQL and a hint set
    output: json plan
    Note: if the SQL execute time exceed statement_timeout setting, it will return statement_timeout + 1s 
    '''
    start = time.time()
    # connect to PG, if failed, exit
    try:
        conn = psycopg2.connect(PG_CONNECTION_STR_TPCH)
    except:
        print("can not connect to PG")
        exit(0)

    cur = conn.cursor()

    for hint in hints:
        cur.execute(hint)
    # execute SQL, if reach the statement_timeout, then the exe time = statement_timeout + 1
    cur.execute(view)
    try:
        cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        explain_json = cur.fetchall()[0][0][0]
        cur.execute(close)
        conn.close()
    except:
        time.sleep(1)
        print('hint to plan error')
    
    stop = time.time()
    return explain_json



def run_query2_hint(view, sql, close, hints):
    '''
    input: a string SQL and two Bao settings
    output: running time of the SQL
    Note: if the SQL execute time exceed statement_timeout setting, it will return statement_timeout + 1s 
    '''
    start = time.time()
    # connect to PG, if failed, exit
    try:
        conn = psycopg2.connect(PG_CONNECTION_STR_TPCH)
    except:
        print("can not connect to PG")
        exit(0)

    cur = conn.cursor()
    for hint in hints:
        cur.execute(hint)
    cur.execute(view)
    try:
        cur.execute("SET statement_timeout TO 3000000") # 3000s 50min 最长
        cur.execute(sql)
        cur.fetchall()
        cur.execute(close)
        conn.close()
    except:
        time.sleep(1)
        print('exe error')
    
    stop = time.time()
    return stop - start


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
    sql_folder = os.path.join(args.data_root_path, args.dataset, '15')
    for query in queries:
        plans = []
        latencys = []
        file_sql_hint_path = os.path.join(args.meta_save_path, f'15_{query[0][0]}.sql.plan')
        file_sql_latency_path = os.path.join(args.meta_save_path, f'15_{query[0][0]}.sql.latency')
        for i in range(NUM_HINT_SET):
            hints = arm_idx_to_hints_v2(i)
            plan = hints_2_plan(query[0][2], query[1][2], query[2][2], hints)
            file_name = os.path.join(sql_folder, f'15_{i}step_{j}.sql')
            if str(plan) in exe_plan_dict.keys():
                latency = exe_plan_dict[str(plan)]
            else:
                latency = run_query2_hint(query[0][2], query[1][2], query[2][2], hints) # default PG
                exe_plan_dict[str(plan)] = latency
            plans.append(plan)
            latencys.append(latency)
            print(f'{query[0][0]}/{len(queries)} query, 15_{query[0][0]} the {i} hint set, execute latency {latency:.5f}s, current {len(exe_plan_dict)} unique plans')
            with open(file_sql_hint_path, 'w') as f:
                json.dump(plans, f)
            with open(file_sql_latency_path, 'w') as f:
                json.dump(latencys, f)
            print('execute finished, the plans and exe latency saved')
    with open(hint_latency_path, 'w') as f:
        json.dump(exe_plan_dict, f)
    print('plan latency dict saved')
    

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print('='*20 + 'start time: ' + nowtime() + '='*20)
    queries = []
    sql_folder = os.path.join(args.data_root_path, args.dataset, '15')
    for i in range(1, 11):
        query = []
        for j in range(3):
            file_name = os.path.join(sql_folder, f'15_{i}step_{j}.sql')
            with open(file_name) as f:
                q = f.read()
            query.append((i, j, q))
        queries.append(query)
    
    main()
    
    t2 = datetime.datetime.now()
    print(''*10 + f'execution cost {t2-t1}')


    print('='*20 + 'end time: ' + nowtime() + '='*20)