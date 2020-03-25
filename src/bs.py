from rqalpha.api import *
import pandas as pd

from const.benchmark import BENCHMARK

# 货币基金
moneyFund = '510880.XSHG'
# 存储数据路径
data_root = "../data/"

bs_dict = {
    "HS300": "601888.XSHG",
    "ZZ500": "300122.XSHE",
    "SZ50": "600519.XSHG"
}


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


def init(context):
    context.init_flag = True
    context.close_df = pd.read_csv(data_root + 'allA_data/allAclose.csv', index_col=0)
    context.rt_df = pd.read_csv(data_root + 'ret/ret_weekly.csv', index_col=0)
    context.bench_root = get_ben_root(context.config.base.benchmark)
    scheduler.run_weekly(trade, tradingday=1)


def trade(context, bar_dict):
    date_data = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
    print("Get data date:", date_data)

    order_target_value(bs_dict[context.bench_root], 50000000)
    ''' 获取基准成分股，用于生成总股池和基准权重，即all_stocks和list_ben_w '''
    # for b in bs_dict:
    #     file = data_root + 'index_weight/%s/%s.csv' % (b, date_data)
    #     df_ben = pd.read_csv(file, encoding='gbk', usecols=['code', 'i_weight'])
    #     df_ben.dropna(inplace=True)
    #     ben_stocks = df_ben['code']
    #     p_df = context.close_df.loc[date_data:, ben_stocks]
    #     ret = p_df.iloc[-1] / p_df.iloc[0] - 1
    #     print(b, ret.argmax())

    # file = data_root + 'index_weight/%s/%s.csv' % (benchmark, date_data)
    # df_ben = pd.read_csv(file, encoding='gbk', usecols=['code', 'i_weight'])
    # df_ben.dropna(inplace=True)
    # ben_stocks = change_code_format_to_long(df_ben['code'])
    #
    # ok_stocks = [s for s in ben_stocks if not is_st_stock(s) and not is_suspended(s)]
    # ret_s = context.rt_df.loc[date_data, ok_stocks]
    # best_stock = ret_s.argmax()
    #
    # for s in pos_stocks:
    #     if s != best_stock and s != moneyFund:
    #         order_target_percent(s, 0)
    #
    # if context.init_flag:
    #     ord = order_target_percent(moneyFund, 0.9)
    #     if ord:
    #         if ord._status == ORDER_STATUS.FILLED:
    #            context.init_flag = False
    #
    # # order_target_percent(best_stock, 0.1)
    # order_target_value(moneyFund, context.stock_account.cash)

