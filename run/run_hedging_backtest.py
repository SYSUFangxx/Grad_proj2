import os

from rqalpha import run_file
from const.benchmark import *
from src.my_calculator import check_root

# s_date = "2016-01-08"
s_date = "2017-01-01"
# e_date = "2018-01-01"
e_date = "2020-01-01"

strategy_path = "../utils/hedging_backtest.py"


def run_file_for_po(play_back_po, play_back_method):
    """
    该函数针对单个优化算法结果做对冲交易回测实验
    :param play_back_po: 回测的投资组合
    :param play_back_method: 回测的优化算法
    :return: 无
    """
    for benchmark, ben_code in BENCHMARK.items():
        res_root = f'../res/backtest/hedging-{play_back_po}-{play_back_method}/{benchmark}'
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
                    "stock": 50000000,
                    "future": 50000000
                }
            },
            "mod":
            {
                "sys_analyser": {
                    "enabled": True,
                    "myParam_po": play_back_po,
                    "myParam_meth": play_back_method,
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


if __name__ == '__main__':
    play_back_pos = ['my_po']
    # play_back_methods = ['calc_weight', 'calc_weight_with_exposure']
    play_back_methods = ['cvxpy_neu', 'cvxpy_with_exposure']
    for po in play_back_pos:
        for method in play_back_methods:
            print('对冲回测：', po, method)
            run_file_for_po(po, method)

