import os

from rqalpha.api import *
from const.benchmark import *
import pandas as pd


def get_ben_root(code):
    for k, v in BENCHMARK.items():
        if v == code:
            return k


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.flag = True

    # 回测的指数
    context.bench_root = get_ben_root(context.config.base.benchmark)

    # """ 对应 run_file_for_po函数 """
    # # 回测的投资组合模型
    # po = context.config.mod.sys_analyser.myParam_po
    # # 回测的优化方法
    # meth = context.config.mod.sys_analyser.myParam_meth
    # # 获取交割单
    # context.trade_df = pd.read_csv(f'../res/backtest/{po}-{meth}/{context.bench_root}/{po}/trades.csv',
    #                                usecols=["last_quantity", "order_book_id", "side", "trading_datetime"])
    # context.trade_df["trading_datetime"] = context.trade_df["trading_datetime"].apply(lambda x: x[:10])

    """ 对应run_for_root函数 """
    # 回测交割单目录，组合优化方法
    context.trade_root = context.config.mod.sys_analyser.myTrade_root
    po = context.config.mod.sys_analyser.myPo.split('-')[0]
    # 获取交割单
    # context.trade_df = pd.read_csv(os.path.join(context.trade_root, f'{context.bench_root}/{po}/trades.csv'),
    #                                usecols=["last_quantity", "order_book_id", "side", "trading_datetime"])
    context.trade_df = pd.read_csv(os.path.join(context.trade_root, f'{context.bench_root}/my_po/trades.csv'),
                                   usecols=["last_quantity", "order_book_id", "side", "trading_datetime"])
    context.trade_df["trading_datetime"] = context.trade_df["trading_datetime"].apply(lambda x: x[:10])

    # 确定运行频率
    # scheduler.run_weekly(rebalance, tradingday=-1)
    # scheduler.run_weekly(my_hedging_test, tradingday=-1)
    scheduler.run_daily(hedging_by_res)


def hedging_by_res(context, bar_dict):
    now_date = context.now.strftime('%Y-%m-%d')
    trade_today = context.trade_df[context.trade_df.trading_datetime == now_date]
    for id, row in trade_today.iterrows():
        # submit_order(row["order_book_id"], row["last_quantity"], row["side"])
        amt = row["last_quantity"] if row["side"] == "BUY" else -row["last_quantity"]
        order_shares(row["order_book_id"], amt)

    """ 做空股指期货 """
    # 多空市值差
    mv_diff = context.stock_account.market_value + context.future_account.market_value  # 做空期货，市值为负数

    if context.bench_root != 'COMPO':
        sell_code = 'IF88' if context.bench_root == 'HS300' else ('IH88' if context.bench_root == 'SZ50' else 'IC88')
        close = bar_dict[sell_code].close
        points = 200 if sell_code == 'IC88' else 300   # 除了中证500是200点，另外两个指数300点
        if mv_diff >= close * points:   # 股票市值大于期货现有市值超过一张合约价值，则做空期货合约
            sell_open(sell_code, mv_diff // (close * points))
        elif mv_diff <= -close * points:    # 股票市值小于期货现有市值超过一张合约价值，则买平期货合约
            buy_close(sell_code, -mv_diff // (close * points))
    else:
        shares = {}
        for sc in ['IF88', 'IH88', 'IC88']:
            close = bar_dict[sc].close
            points = 200 if sc == 'IC88' else 300
            shares[sc] = mv_diff / abs(mv_diff) * (abs(mv_diff) / 3 // (close * points))
        if shares['IF88'] > 0 and shares['IH88'] > 0 and shares['IC88'] > 0:
            sell_open('IF88', shares['IF88'])
            sell_open('IH88', shares['IH88'])
            sell_open('IC88', shares['IC88'])
        if shares['IF88'] < 0 and shares['IH88'] < 0 and shares['IC88'] < 0:
            buy_close('IF88', -shares['IF88'])
            buy_close('IH88', -shares['IH88'])
            buy_close('IC88', -shares['IC88'])

