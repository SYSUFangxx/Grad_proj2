from rqalpha import run_file
from const.benchmark import BENCHMARK
from const.my_po_params import MY_PO_PARAMS
from src.my_calculator import check_root
import os

# s_date = "2016-01-08"
s_date = "2017-01-01"
# e_date = "2018-01-01"
# e_date = "2017-01-03"
e_date = "2019-01-01"
config_cw = {
    "upper": MY_PO_PARAMS.WEIGHT_UPPER,
    "c": MY_PO_PARAMS.TRANSACTION_COST,
    "l": MY_PO_PARAMS.L,
    "lam": MY_PO_PARAMS.LAM,
    "exposure": MY_PO_PARAMS.EXPOSURE,
    "opt_tool_method": MY_PO_PARAMS.opt_tool_method
}

strategy_name = 'my_po'
strategy_path = "../src/%s.py" % strategy_name

# 保存优化问题的各个变量和解向量
op_file = open("../res/opt_weight/my_po_opt_tool.txt", 'w')

for benchmark, ben_code in BENCHMARK.items():
    res_root = '../res/backtest/%s/%s' % (strategy_name + "-" + config_cw["opt_tool_method"], benchmark)
    check_root('/'.join(res_root.split('/')[:3]))
    check_root('/'.join(res_root.split('/')[:4]))
    check_root(res_root)

    config = {
        "extra": {
            "context_vars": {
                "CONFIG_CW": config_cw,
                "CALC_WEIGHT": config_cw["opt_tool_method"],
                "op_file": op_file
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

    print(f'\n{benchmark}\n')
