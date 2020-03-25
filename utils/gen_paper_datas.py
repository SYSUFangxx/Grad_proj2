import os

import pandas as pd
import numpy as np

""" 该程序主要目的是：统计生成论文所需要用到的数据、图、表 """


def get_diff_risk_data(neu_root='../res/backtest/my_po-calc_weight',
                       expo_root='../res/backtest/my_po-calc_weight_with_exposure'):
    """
    获得不同风险约束的统计指标，包括CW（累积财富）、α、MDD（最大回撤）、SR（夏普比率）
    :param neu_root: 风险中性组合文件目录
    :param expo_root: 适度风险暴露组合文件目录
    :return: 一个三维的DataFrame。三个维度分别是：指标、数据集、方法
    """
    inds = [np.array(['HS300', 'HS300', 'SZ50', 'SZ50', 'ZZ500', 'ZZ500']),
            np.array(['MY_PO', 'MY_PO_EXPO', 'MY_PO', 'MY_PO_EXPO', 'MY_PO', 'MY_PO_EXPO'])]

    cols = np.array(['CW', 'α', 'MDD', 'SR'])

    res_df = pd.DataFrame(index=inds, columns=cols)
    for bench in ['HS300', 'SZ50', 'ZZ500']:
        for root, po in zip([neu_root, expo_root], ['MY_PO', 'MY_PO_EXPO']):
            df = pd.read_csv(os.path.join(root, bench, 'my_po', 'summary.csv'), index_col=0)
            res_df.loc[(bench, po), :] = df.loc[['unit_net_value', 'alpha', 'max_drawdown', 'sharpe']].transpose().values

    return res_df


def gen_datas():
    # 获取不同风险约束的统计指标
    neu_root = '../res/backtest/my_po-cvxpy_neu'
    expo_root = '../res/backtest/my_po-cvxpy_with_exposure'
    df = get_diff_risk_data(neu_root, expo_root)
    df.to_csv('../res/paper_datas/neu_vs_expo.csv')


if __name__ == '__main__':
    gen_datas()
