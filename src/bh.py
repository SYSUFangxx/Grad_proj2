import os

from rqalpha.api import *
import pandas as pd
import numpy as np

from const.special_codes import EXCLUDE_STOCKS
from const.benchmark import BENCHMARK

# 去除股票
exclude_stocks = EXCLUDE_STOCKS
# 货币基金
moneyFund = '510880.XSHG'
# 存储数据路径
data_root = "../data/"


# @assistant function
#  return the format of rqalpha
def change_code_format_to_long(stocks):
    stocks = [s.replace('SH', 'XSHG') for s in stocks]
    stocks = [s.replace('XXSHGG', 'XSHG') for s in stocks]
    stocks = [s.replace('SZ', 'XSHE') for s in stocks]
    return stocks


def change_code_format_to_short(stocks):
    stocks = [s.replace('XSHG', 'SH') for s in stocks]
    stocks = [s.replace('XSHE', 'SZ') for s in stocks]
    return stocks


def get_ben_root(code):
    for k, v in BENCHMARK.items():
        if v == code:
            return k


def adjust_weights_to_pos(w):
    """
    将组合权重向量中的负数置为0，并对正数进行归一化操作
    :param w: 组合权重向量
    :return: 调整后的组合权重
    """
    for i in range(len(w)):
        if w[i] < 0:
            w[i] = 0
    return w / np.sum(w)


def init(context):
    context.init_flag = True
    context.bench_root = get_ben_root(context.config.base.benchmark)
    scheduler.run_weekly(trade, tradingday=1)


def trade(context, bar_dict):
    date_data = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
    print("Get data date:", date_data)

    if context.init_flag:
        # 基准成分股，生成成分股权重s_ben_weights、基准成分股ben_stocks
        index_file_csv = sorted([p for p in os.listdir(data_root + 'index_weight/%s' % context.bench_root)
                                 if p.split('.')[0] <= date_data]
                                )[-1]   # 取小于等于date_data的最大日期文件
        file = data_root + 'index_weight/%s/%s' % (context.bench_root, index_file_csv)
        df_ben = pd.read_csv(file, encoding='gbk', usecols=['con_code', 'weight'])
        df_ben.dropna(inplace=True)
        drop_stocks = [s for s in change_code_format_to_long(df_ben.con_code) if is_suspended(s)]
        df_ben.index = df_ben.con_code
        df_ben = df_ben.drop(change_code_format_to_short(drop_stocks))
        ben_stocks = change_code_format_to_long(df_ben.con_code)
        ben_weight = df_ben.weight / 100
        ben_weight = adjust_weights_to_pos(ben_weight)
        for s, w in zip(ben_stocks, ben_weight):
            if s in exclude_stocks:
                continue
            order_target_percent(s, w)
        context.init_flag = False
    # if context.stock_account.cash > 0:
    #     order_target_value(moneyFund, context.stock_account.cash)

