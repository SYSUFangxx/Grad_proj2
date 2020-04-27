from rqalpha import run_file
from const.benchmark import BENCHMARK
from src.my_calculator import check_root
import os

s_date = "2017-01-01"
e_date = "2020-01-01"
other_pos = ['bh', 'ubh', 'cr', 'semi_cr', 'eg', 'olu', 'olmar', 'rmr', 'sspo']
have_run = ['bh', 'ubh', 'cr', 'semi_cr', 'eg', 'olu', 'rmr', 'sspo']

for po in other_pos:
    if po in have_run:
        continue

    strategy_path = "../src/%s.py" % po
    check_root('../res/backtest/%s' % po)

    for benchmark, ben_code in BENCHMARK.items():
        # if benchmark == 'HS300': continue
        print('Algorithm:\t', po, 'a\tBenchmark:\t', benchmark)

        res_root = '../res/backtest/%s/%s' % (po, benchmark)
        check_root(res_root)

        config = {
            "base":
            {
                "start_date": s_date,
                "end_date": e_date,
                "benchmark": ben_code,
                "accounts": {
                    "stock": 50000000
                }
            },
            "mod":
            {
                "sys_analyser": {
                    "enabled": True,
                    "plot": False,
                    "plot_save_file": res_root,
                    "report_save_path": res_root,
                    "output_file": os.path.join(res_root, "result.pkl")
                },
                "sys_simulation": {
                    "slippage": 0.002
                }
            }
        }

        run_file(strategy_path, config)
