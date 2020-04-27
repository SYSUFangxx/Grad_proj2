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
        res_root = '../res/backtest/hedging-%s/%s' % (
            play_back_po + ('-' +  play_back_method if play_back_po == 'my_po' else ''), benchmark)
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
                    "myRun_func": "run_file_for_po",
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


def run_for_root(roots):
    """
    该函数针对多个优化算法的结果文件夹做对冲交易回测实验
    :param roots: 需要回撤的结果文件夹，对应下层为指数文件夹
    :return: 无
    """
    for root in roots:
        for benchmark, ben_code in BENCHMARK.items():
            if benchmark != 'COMPO': continue

            po = root.split('/')[-1]
            print(f"{root}\t\t{po}\t\t{benchmark}")

            res_root = f'../res/backtest_hedging/{po}/{benchmark}'
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
                        "future": 100000000
                    }
                },
                "mod":
                {
                    "sys_analyser": {
                        "enabled": True,
                        "myPo": po,
                        "myTrade_root": root,
                        "myRun_func": "run_for_root",
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
    # """ 对冲实验：针对不同模型、算法"""
    # play_back_pos = ['my_po']
    # # play_back_methods = ['calc_weight', 'calc_weight_with_exposure']
    # # play_back_methods = ['cvxpy_neu', 'cvxpy_with_exposure']
    # play_back_methods = ['cvxpy_neu']
    # for po in play_back_pos:
    #     for method in play_back_methods:
    #         print('对冲回测：', po, method)
    #         run_file_for_po(po, method)

    """ 对冲实验：针对不同模型、算法"""
    # res_roots = ['../backup/res20200415/' + r for r in ['res_neu', 'res_exposure']]
    # res_roots = ['../res/backtest/my_po-cvxpy_neu', '../res/backtest/my_po-cvxpy_with_exposure']
    res_roots = ['']
    run_for_root(res_roots)
