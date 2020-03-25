from rqalpha import run_file
from const.benchmark import BENCHMARK
from src.my_calculator import check_root
import os

# s_date = "2016-01-08"
s_date = "2017-01-01"
# e_date = "2018-01-01"
e_date = "2020-01-01"

strategys = ['bh', 'cr', 'ubh', 'semi_cr']
# strategys = ['eg', 'olu', 'olmar', 'sspo', 'rmr']
# strategys = ['eg']
# strategys = ['sspo']
# strategys = ['olu']

for ist in strategys:
    strategy_path = "../src/%s.py" % ist

    check_root('../res/backtest/%s' % ist)

    for benchmark, ben_code in BENCHMARK.items():
        res_root = '../res/backtest/%s/%s' % (ist, benchmark)
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
