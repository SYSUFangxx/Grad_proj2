import os

from rqalpha import run_file
from const.benchmark import BENCHMARK
from src.my_calculator import check_root

# s_date = "2015-12-31"
s_date = "2016-01-05"
e_date = "2020-01-01"

strategy_name = 'my_po'
strategy_path = "../utils/empty_strategy.py"

for benchmark, ben_code in BENCHMARK.items():
    res_root = '../res/dataset/%s' % benchmark
    check_root('/'.join(res_root.split('/')[:3]))
    check_root('/'.join(res_root.split('/')[:4]))
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

    print(f'\n{benchmark}\n')
