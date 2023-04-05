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
'''some tools for db experiment
execute a query
'''



import psycopg2
import time
import os

PG_CONNECTION_STR_JOB = "dbname=imdbload user=aaa"
PG_CONNECTION_STR_TPCH = "dbname=tpchload user=aaa"

_ALL_OPTIONS = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
]


# https://rmarcus.info/appendix.html
all_48_hint_sets = '''hashjoin,indexonlyscan
hashjoin,indexonlyscan,indexscan
hashjoin,indexonlyscan,indexscan,mergejoin
hashjoin,indexonlyscan,indexscan,mergejoin,nestloop
hashjoin,indexonlyscan,indexscan,mergejoin,seqscan
hashjoin,indexonlyscan,indexscan,nestloop
hashjoin,indexonlyscan,indexscan,nestloop,seqscan
hashjoin,indexonlyscan,indexscan,seqscan
hashjoin,indexonlyscan,mergejoin
hashjoin,indexonlyscan,mergejoin,nestloop
hashjoin,indexonlyscan,mergejoin,nestloop,seqscan
hashjoin,indexonlyscan,mergejoin,seqscan
hashjoin,indexonlyscan,nestloop
hashjoin,indexonlyscan,nestloop,seqscan
hashjoin,indexonlyscan,seqscan
hashjoin,indexscan
hashjoin,indexscan,mergejoin
hashjoin,indexscan,mergejoin,nestloop
hashjoin,indexscan,mergejoin,nestloop,seqscan
hashjoin,indexscan,mergejoin,seqscan
hashjoin,indexscan,nestloop
hashjoin,indexscan,nestloop,seqscan
hashjoin,indexscan,seqscan
hashjoin,mergejoin,nestloop,seqscan
hashjoin,mergejoin,seqscan
hashjoin,nestloop,seqscan
hashjoin,seqscan
indexonlyscan,indexscan,mergejoin
indexonlyscan,indexscan,mergejoin,nestloop
indexonlyscan,indexscan,mergejoin,nestloop,seqscan
indexonlyscan,indexscan,mergejoin,seqscan
indexonlyscan,indexscan,nestloop
indexonlyscan,indexscan,nestloop,seqscan
indexonlyscan,mergejoin
indexonlyscan,mergejoin,nestloop
indexonlyscan,mergejoin,nestloop,seqscan
indexonlyscan,mergejoin,seqscan
indexonlyscan,nestloop
indexonlyscan,nestloop,seqscan
indexscan,mergejoin
indexscan,mergejoin,nestloop
indexscan,mergejoin,nestloop,seqscan
indexscan,mergejoin,seqscan
indexscan,nestloop
indexscan,nestloop,seqscan
mergejoin,nestloop,seqscan
mergejoin,seqscan
nestloop,seqscan'''


all_48_hint_sets = all_48_hint_sets.split('\n')
all_48_hint_sets = [ ["enable_"+j for j in i.split(',')] for i in all_48_hint_sets]
# print([len(i) for i in all_48_hint_sets])

def arm_idx_to_hints(arm_idx):
    hints = []
    for option in _ALL_OPTIONS:
        hints.append(f"SET {option} TO off")

    if arm_idx == 0:
        for option in _ALL_OPTIONS:
            hints.append(f"SET {option} TO on") # default PG setting 
    elif arm_idx == 1:
        hints.append("SET enable_hashjoin TO on")
        hints.append("SET enable_indexonlyscan TO on")
        hints.append("SET enable_indexscan TO on")
        hints.append("SET enable_mergejoin TO on")
        hints.append("SET enable_seqscan TO on")
    elif arm_idx == 2:
        hints.append("SET enable_hashjoin TO on")
        hints.append("SET enable_indexonlyscan TO on")
        hints.append("SET enable_nestloop TO on")
        hints.append("SET enable_seqscan TO on")
    elif arm_idx == 3:
        hints.append("SET enable_hashjoin TO on")
        hints.append("SET enable_indexonlyscan TO on")
        hints.append("SET enable_seqscan TO on")
    elif arm_idx == 4:
        hints.append("SET enable_hashjoin TO on")
        hints.append("SET enable_indexonlyscan TO on")
        hints.append("SET enable_indexscan TO on")
        hints.append("SET enable_nestloop TO on")
        hints.append("SET enable_seqscan TO on")
    else:
        print('5 hint set error')
        exit(0)
    return hints

def arm_idx_to_hints_v2(arm_idx):
    hints = []
    for option in _ALL_OPTIONS:
        hints.append(f"SET {option} TO off")

    if arm_idx > -1 and arm_idx < 48:
        for i in all_48_hint_sets[arm_idx]:
            hints.append(f"SET {i} TO on")

    elif arm_idx == 48:
        for option in _ALL_OPTIONS:
            hints.append(f"SET {option} TO on") # default PG setting 
    else:
        print('48 hint set error')
        exit(0)
    return hints



def run_query(sql):
    '''
    input: a string SQL and two Bao settings
    output: running time of the SQL
    Note: if the SQL execute time exceed statement_timeout setting, it will return statement_timeout + 1s 
    '''
    start = time.time()
    # connect to PG, if failed, exit
    try:
        conn = psycopg2.connect(PG_CONNECTION_STR_JOB)
    except:
        print("can not connect to PG")
        exit(0)

    cur = conn.cursor()
    # execute SQL, if reach the statement_timeout, then the exe time = statement_timeout + 1
    try:
        cur.execute("SET statement_timeout TO 3000000") # 3000s 50min 最长
        cur.execute(sql)
        cur.fetchall()
        conn.close()
    except:
        time.sleep(1)
        print('exe error')
    
    stop = time.time()
    return stop - start


def run_query_hint(sql, hints):
    '''
    input: a string SQL and two Bao settings
    output: running time of the SQL
    Note: if the SQL execute time exceed statement_timeout setting, it will return statement_timeout + 1s 
    '''
    start = time.time()
    # connect to PG, if failed, exit
    try:
        conn = psycopg2.connect(PG_CONNECTION_STR_JOB)
    except:
        print("can not connect to PG")
        exit(0)

    cur = conn.cursor()
    for hint in hints:
        cur.execute(hint)
    # execute SQL, if reach the statement_timeout, then the exe time = statement_timeout + 1
    try:
        cur.execute("SET statement_timeout TO 3000000") # 3000s 50min 最长
        cur.execute(sql)
        cur.fetchall()
        conn.close()
    except:
        time.sleep(1)
        print('exe error')
    
    stop = time.time()
    return stop - start





def hints_to_plan(sql, hints):
    '''
    input: a string SQL and a hint set
    output: json plan
    Note: if the SQL execute time exceed statement_timeout setting, it will return statement_timeout + 1s 
    '''
    start = time.time()
    # connect to PG, if failed, exit
    try:
        conn = psycopg2.connect(PG_CONNECTION_STR_JOB)
    except:
        print("can not connect to PG")
        exit(0)

    cur = conn.cursor()

    for hint in hints:
        cur.execute(hint)
    # print('all hints are set')
    # execute SQL, if reach the statement_timeout, then the exe time = statement_timeout + 1
    try:
        cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        # cur.execute(sql)
        # print('execute success')
        explain_json = cur.fetchall()[0][0][0]
        # print('fetch succ', explain_json)
        # bao_props, _qplan = explain_json
        # print(bao_props)
        # bao_plan = json.loads(bao_props["Bao"]["Bao plan JSON"])
        # bao_buffer = json.loads(bao_props["Bao"]["Bao buffer JSON"])
        conn.close()
    except:
        time.sleep(1)
        print('hint to plan error')
    
    stop = time.time()
    return explain_json


def sql_to_cost(sql, hints=None):
    '''
    input: a string SQL and a hint set
    output: json plan
    Note: if the SQL execute time exceed statement_timeout setting, it will return statement_timeout + 1s 
    '''
    start = time.time()
    # connect to PG, if failed, exit
    try:
        conn = psycopg2.connect(PG_CONNECTION_STR_JOB)
    except:
        print("can not connect to PG")
        exit(0)

    cur = conn.cursor()
    if hints is not None:
        for hint in hints:
            cur.execute(hint)
    # print('all hints are set')
    # execute SQL, if reach the statement_timeout, then the exe time = statement_timeout + 1
    try:
        cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        explain_json = cur.fetchall()[0][0][0]
        conn.close()
    except:
        time.sleep(1)
        print('sql to cost error')
    
    stop = time.time()
    return explain_json['Plan']['Total Cost']

def read_dataset_queries(args):
    queries = []
    sql_folder = os.path.join(args.data_root_path, args.dataset)
    sql_files = os.listdir(sql_folder)
    for fp in sql_files:
        if fp.split('.')[-1] == 'sql':
            fp_full = os.path.join(sql_folder, fp)
            with open(fp_full) as f:
                query = f.read()
            queries.append((fp, query)) # (1a, slelect ... )
    print(f'read {len(queries)} queries from {sql_folder}')
    return queries


