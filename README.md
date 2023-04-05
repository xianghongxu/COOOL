This is the implementation of COOOL. In this repository, we show the saved models as well as the raw data for all experiments involved in the paper.
# Requirements
- Install PostgreSQL version 12.5
```bash
# Download link
wget https://ftp.postgresql.org/pub/source/v12.5/postgresql-12.5.tar.gz
```
- The configuration of PostgreSQL is shown in 'postgresql.conf', which is tuned by [PGTune](https://pgtune.leopard.in.ua/#/).

- Prepare datasets and workloads.

Due to license issues, we cannot provide queries in this repository. All queries of the two datasets should be put under `db_benchmark_datasets/` folder. Refer to `https://github.com/gregrahn/join-order-benchmark` and `https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp` for join-order-benchmark and TPC-H, respectively. We omit the details of how to load data into PostgreSQL.

- Execute queries with each hint set and record the experience.
    - Change the dbname and user in `src/db_util.py` with your own settings.
    - Execute `python split15.py` to split CREATE VIEW and SELECT operation in TPC-H template #15, because it is not applicable to directly get the plan of the query with operation 'CREATE VIEW'.
    - Run `python execute_queries.py --dataset='join-order-benchmark'` to execute and record the execution experimence of join-order-benchmark.
    - Run `python execute_queries.py --dataset='TPCH'` and `python execute_15.py` to execute and record the execution experimence of TPC-H.

In order to facilitate reproduction, we have stored the results of execution on our machine under `record/postgresql/{dataset}/execute_sql/`.

- Python 3.8 with the following packages
```
torch==1.12.0 (w/ cuda, refer to https://download.pytorch.org/whl/torch/)
sklearn==0.0.post1
scipy==1.9.3
scikit-learn==1.2.0
psycopg2-binary==2.9.5
pandas==1.5.2
numpy==1.22.0
```

# Obtain the reported results in the paper
The raw data (training logs) are shown in the corresponding folders under `record/postgresql`. We run each experiments 10 times, so all experiments have 10 saved models. To obtain the repored results in the paper, we can directly analyze from the raw data.
```python
python ana_single.py # the single instance experiments
python ana_cross.py # cross workload experiments
python ana_unione.py # unified model experiments
```
It is also available for running the inference stages of all models.
```bash
bash inference.sh
```

This script will generate the inference logs of each experimental settings by execute the saved models. There are 60 experimental settings, so there are 60 inference logs and 600 saved models will be executed. We use some examples to illustrate the meaning of the arguments, the commands of all experiments are shown in `inference.sh`

```python
# inference in the single instance setting
python3 -u run.py  
    --exp='single' # ['single', 'cross', 'one'], three experimental settings
    --app='pair' # ['Bao', 'pair', 'list'], three approaches involved in the paper
    --grain='adhoc' # ['adhoc', 'repeat'], two experimental scenarios
    --dataset='TPCH' # ['join-order-benchmark', 'TPCH'], two datasets
    --slow=1 # [0, 1], slow split or not

# inference in the cross workload setting
python3 -u run.py 
    --exp='cross' # cross workload setting
    --src='TPCH' # source workload
    --tgt='JOB'  # target workload, 'join-order-benchmark' and 'JOB' are equivalent
    --app='pair' # the pairwise approach
    --grain='adhoc'  # adhoc scenario
    --slow=1 # slow split

# inference in the unified model setting
python3 -u run.py  
    --exp='one' # unified model setting
    --app='pair' # the pairwise approach
    --grain='adhoc'  # adhoc scenario
    --slow=1 # slow split
    

```