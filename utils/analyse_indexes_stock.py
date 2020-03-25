import pandas as pd
import os

""" 该程序的主要目的是：分析每个指数包含优质股的情况 """


def stat_index_gc(index):
    """
    统计全部指数成分股每17周的收益情况，并写入结果文件中
    :param index: 股票指数
    :return: 无
    """
    paths = os.listdir(f'../data/index_weight/{index}')
    stks = set()
    for p in paths:
        df = pd.read_csv(f'../data/index_weight/{index}/{p}')
        for c in df.con_code:
            stks.add(c)
    stks = sorted(stks)
    df_ret = pd.read_csv('../data/ret/ret_17week.csv', index_col=0, encoding='gbk')
    stks_miss = [s for s in stks if s not in df_ret.columns]
    print(f"缺失收益率数据：{stks_miss}")
    stks = [s for s in stks if s not in stks_miss]
    df_ret[stks].to_csv(f'../data/analysis/{index}-gc.csv')


if __name__ == '__main__':
    for ind in ['HS300', 'SZ50', 'ZZ500']:
        print(ind)
        stat_index_gc(ind)
