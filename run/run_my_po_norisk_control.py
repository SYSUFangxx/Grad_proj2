from rqalpha import run_file
from datetime import datetime
import os

from const.benchmark import BENCHMARK
from const.my_po_params import MY_PO_PARAMS
from src.my_calculator import check_root

# s_date = "2016-01-08"
s_date = "2017-01-01"
# e_date = "2018-01-01"
# e_date = "2019-01-01"
e_date = "2020-01-01"

# 考虑同时回测风险中性和风险暴露
# for meth in ["calc_weight", "calc_weight_with_exposure"]:
for meth in ["cvxpy_without_risk_control"]:
    config_cw = {
        "upper": MY_PO_PARAMS.WEIGHT_UPPER,
        "c": MY_PO_PARAMS.TRANSACTION_COST,
        "l": MY_PO_PARAMS.L,
        "lam": MY_PO_PARAMS.LAM,
        "exposure": MY_PO_PARAMS.EXPOSURE,
        # "calc_method": MY_PO_PARAMS.CALC_METHOD
        "calc_method": meth
    }

    strategy_name = 'my_po_norisk_control'
    strategy_path = "../src/%s.py" % strategy_name

    # 保存优化问题的各个变量和解向量,每一行的格式为{'index_name':{'backtest_date': {key: value}}}
    opt_file = open(f"../res/opt_weight/my_po-{config_cw['calc_method']}-{str(datetime.now().strftime('%Y%m%d'))}.txt",
                    'w')

    for benchmark, ben_code in BENCHMARK.items():
        # if benchmark != 'ZZ500': continue

        print("=" * 40 + "\n" + f"method:\t{meth}\nbenchmark:\t{benchmark}\n" + "=" * 40)

        res_root = '../res/backtest/%s/%s' % (strategy_name + "-" + config_cw["calc_method"], benchmark)
        check_root('/'.join(res_root.split('/')[:3]))
        check_root('/'.join(res_root.split('/')[:4]))
        check_root(res_root)

        config = {
            "extra": {
                "context_vars": {
                    "CONFIG_CW": config_cw,
                    "bench_record": benchmark,
                    "opt_file": opt_file
                },
                # "log_level": "error",
            },
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
