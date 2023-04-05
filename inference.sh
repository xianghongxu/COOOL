#===========single instance adhoc: 3 apps x slow or not x 2 datasets=12


python3 -u run.py  --app='Bao' --grain='adhoc' --exp='single'


python3 -u run.py  --app='pair' --grain='adhoc' --exp='single'


python3 -u run.py  --app='list' --grain='adhoc' --epochs=200  --exp='single'


python3 -u run.py  --app='Bao' --grain='adhoc' --slow=1 --exp='single'


python3 -u run.py  --app='pair' --grain='adhoc' --slow=1 --exp='single'


python3 -u run.py  --app='list' --grain='adhoc' --slow=1 --epochs=200  --exp='single'



python3 -u run.py  --app='Bao' --grain='adhoc' --dataset='TPCH' --exp='single'




python3 -u run.py  --app='pair' --grain='adhoc' --dataset='TPCH' --exp='single'




python3 -u run.py  --app='list' --grain='adhoc' --dataset='TPCH' --epochs=200  --exp='single'




python3 -u run.py  --app='Bao' --grain='adhoc' --slow=1 --dataset='TPCH' --exp='single'




python3 -u run.py  --app='pair' --grain='adhoc' --slow=1 --dataset='TPCH' --exp='single'




python3 -u run.py  --app='list' --grain='adhoc' --slow=1 --dataset='TPCH' --epochs=200  --exp='single'


#===========single instance repeat: 3 apps x slow or not x 2 datasets=12


python3 -u run.py  --app='Bao' --grain='repeat' --exp='single'




python3 -u run.py  --app='pair' --grain='repeat' --exp='single'




python3 -u run.py  --app='list' --grain='repeat' --epochs=200  --exp='single'




python3 -u run.py  --app='Bao' --grain='repeat' --slow=1 --exp='single'




python3 -u run.py  --app='pair' --grain='repeat' --slow=1 --exp='single'




python3 -u run.py  --app='list' --grain='repeat' --slow=1 --epochs=200  --exp='single'




python3 -u run.py  --app='Bao' --grain='repeat' --dataset='TPCH' --exp='single'




python3 -u run.py  --app='pair' --grain='repeat' --dataset='TPCH' --exp='single'




python3 -u run.py  --app='list' --grain='repeat' --dataset='TPCH' --epochs=200  --exp='single'




python3 -u run.py  --app='Bao' --grain='repeat' --slow=1 --dataset='TPCH' --exp='single'




python3 -u run.py  --app='pair' --grain='repeat' --slow=1 --dataset='TPCH' --exp='single'




python3 -u run.py  --app='list' --grain='repeat' --slow=1 --dataset='TPCH' --epochs=200  --exp='single'


#===========cross workload: 3 apps x slow or not x T-J J-T x repeat adhoc=24


python3 -u run.py --src='JOB' --tgt='TPCH'  --app='Bao' --grain='adhoc' --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='Bao' --grain='adhoc' --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='Bao' --grain='repeat' --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='Bao' --grain='repeat' --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='Bao' --grain='adhoc' --slow=1 --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='Bao' --grain='adhoc' --slow=1 --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='Bao' --grain='repeat' --slow=1 --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='Bao' --grain='repeat' --slow=1 --exp='cross'





python3 -u run.py --src='JOB' --tgt='TPCH'  --app='list' --grain='adhoc' --epochs=200  --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='list' --grain='adhoc' --epochs=200  --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='list' --grain='repeat' --epochs=200  --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='list' --grain='repeat' --epochs=200  --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='list' --grain='adhoc' --epochs=200 --slow=1 --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='list' --grain='adhoc' --epochs=200  --slow=1 --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='list' --grain='repeat' --epochs=200 --slow=1 --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='list' --grain='repeat' --epochs=200 --slow=1 --exp='cross'





python3 -u run.py --src='JOB' --tgt='TPCH'  --app='pair' --grain='adhoc' --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='pair' --grain='adhoc' --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='pair' --grain='repeat' --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='pair' --grain='repeat' --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='pair' --grain='adhoc' --slow=1 --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='pair' --grain='adhoc' --slow=1 --exp='cross'




python3 -u run.py --src='JOB' --tgt='TPCH'  --app='pair' --grain='repeat' --slow=1 --exp='cross'




python3 -u run.py --src='TPCH' --tgt='JOB'  --app='pair' --grain='repeat' --slow=1 --exp='cross'


#===========one model: 3 apps x slow or not x  repeat adhoc=12


python3 -u run.py  --app='list' --grain='adhoc' --epochs=200 --exp='one'




python3 -u run.py  --app='list' --grain='adhoc' --epochs=200 --slow=1 --exp='one'




python3 -u run.py  --app='list' --grain='repeat' --epochs=200 --exp='one'




python3 -u run.py  --app='list' --grain='repeat' --epochs=200 --slow=1 --exp='one'




python3 -u run.py  --app='Bao' --grain='adhoc' --exp='one'




python3 -u run.py  --app='Bao' --grain='adhoc'  --slow=1 --exp='one'




python3 -u run.py  --app='Bao' --grain='repeat'  --exp='one'




python3 -u run.py  --app='Bao' --grain='repeat' --slow=1 --exp='one'




python3 -u run.py  --app='pair' --grain='adhoc' --exp='one'




python3 -u run.py  --app='pair' --grain='adhoc'  --slow=1 --exp='one'




python3 -u run.py  --app='pair' --grain='repeat'  --exp='one'




python3 -u run.py  --app='pair' --grain='repeat' --slow=1 --exp='one'

