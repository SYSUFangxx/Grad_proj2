import os

import pandas as pd
import numpy as np

""" 该程序主要目的是：统计生成论文所需要用到的数据、图、表 """


def plot_benchmark(src_root='../res/dataset/', res_root='../res/paper_datas'):
    """
    绘制基准在这几年的单位净值变化，净值保存为.csv文件，并绘制净值变化图
    :param src_root: 数据目录，可以是任一包含2016-2020时间区间的回测结果。这里采用的是Grad_proj的无操作回测结果
    :return: 无
    """
    from matplotlib import rcParams, pyplot as plt
    # 正常显示中文
    rcParams['font.sans-serif'] = ['SimHei'] + rcParams['font.sans-serif']

    df = pd.DataFrame()
    for ir in ['HS300', 'SZ50', 'ZZ500']:
        root = src_root + ir
        result_dict = pd.read_pickle(os.path.join(root, "result.pkl"))
        benchmark_portfolio = result_dict.get("benchmark_portfolio")
        bv = benchmark_portfolio["unit_net_value"]
        df[ir] = bv
    df.to_csv(os.path.join(src_root, 'benchmark_net_value.csv'))
    df.plot()
    plt.xlabel('date', fontsize='xx-large')
    plt.ylabel('unit net value', fontsize='xx-large')
    plt.show()
    # plt.savefig(os.path.join(res_root, '4_1_benchmark.png'))


def gen_benchmark_info():
    """
    该函数统计指数四年间一共有几只股票等信息
    :return: 无
    """
    root = '../data/index_weight/'
    benchmarks = ['COMPO', 'HS300', 'SZ50', 'ZZ500']
    dict_res = {}
    for bench in benchmarks:
        dict_res[bench] = []
        paths = os.listdir(root + bench)
        for p in paths:
            df = pd.read_csv(root + bench + '/' + p)
            dict_res[bench] += (df['con_code'].tolist())
        dict_res[bench] = sorted(set(dict_res[bench]))
    print([(b, len(s)) for b, s in dict_res.items()])


def plot_diff_cw():
    """
    该函数为了绘制论文中不同算法的累积财富趋势图
    :return: 无
    """
    import matplotlib.pyplot as plt
    benchmarks = ['HS300', 'SZ50', 'ZZ500']
    pos = ['my_po', 'bh', 'cr', 'eg', 'olmar', 'olu', 'rmr', 'semi_cr', 'sspo', 'ubh']
    colors = ['r', 'orange', 'g', 'b', 'y', 'c', 'pink', 'm', 'gold', 'royalblue', 'k', 'chocolate', 'deeppink']

    for bench in benchmarks:
        df = pd.DataFrame()
        for i, s in enumerate(pos):
            # result_pickle_file_path = os.path.join(f'../res/backtest/{s}/{bench}/result.pkl')
            result_pickle_file_path = os.path.join(f'../backup/res20200424-other/{s}/{bench}/result.pkl')
            result_dict = pd.read_pickle(result_pickle_file_path)
            df_port = result_dict['portfolio']
            df[s] = df_port['unit_net_value']
            if i == len(pos) - 1:
                df['market'] = result_dict.get("benchmark_portfolio")['unit_net_value']
        # df.plot()
        df.plot(color=colors)
        plt.title(bench)
        plt.xlabel('Date', fontsize='xx-large')
        plt.ylabel('Cumulative Wealth', fontsize='xx-large')
        plt.show()


def gen_pos_perform():
    """
    该函数为了统计论文中不同算法对比的指标，包括累积财富、alpha值、最大回撤和夏普比率
    :return: 无，直接写入结果文件
    """
    pos = ['bh', 'cr', 'eg', 'olmar', 'olu', 'rmr', 'semi_cr', 'sspo', 'ubh', 'my_po']
    indexs = ['unit_net_value', 'alpha', 'sharpe', 'max_drawdown']
    benchmarks = ['HS300', 'SZ50', 'ZZ500']

    dict_res = {}
    for bench in benchmarks:
        dict_res[bench] = pd.DataFrame()
        for po in pos:
            df = pd.read_csv(f'../res/backtest/{po}/{bench}/{po}/summary.csv', index_col=0)
            dict_res[bench][po] = df.loc[indexs, :].iloc[:, 0]

    for i, idx in enumerate(indexs):
        df = pd.DataFrame()
        for bench in benchmarks:
            df[bench] = dict_res[bench].loc[idx, :]
        df.to_csv(f'../res/paper_datas/4_{i + 2}_{idx}.csv')


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


if __name__ == '__main__':
    # # 绘制基准单位净值曲线
    # plot_benchmark()
    # # 获取基准指数的基本信息
    # gen_benchmark_info()

    # # 获取不同风险约束的统计指标
    # neu_root = '../backup/res20200415/res_neu'
    # expo_root = '../backup/res20200415/res_exposure'
    # df = get_diff_risk_data(neu_root, expo_root)
    # df.to_csv('../res/paper_datas/neu_vs_expo.csv')

    # 获取不同算法的指标对比
    # gen_pos_perform()

    # 绘制不同算法的累积财富曲线
    plot_diff_cw()
