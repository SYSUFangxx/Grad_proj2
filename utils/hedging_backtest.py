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
    # 回测的投资组合模型
    po = context.config.mod.sys_analyser.myParam_po
    # 回测的优化方法
    meth = context.config.mod.sys_analyser.myParam_meth
    # 回测的指数
    context.bench_root = get_ben_root(context.config.base.benchmark)
    # 获取交割单
    context.trade_df = pd.read_csv(f'../res/backtest/{po}-{meth}/{context.bench_root}/{po}/trades.csv',
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

    if context.flag:
        sell_codes = 'IF88' if context.bench_root == 'HS300' else ('IH88' if context.bench_root == 'SZ50' else 'IC88')
        sell_open(sell_codes, 50)
        context.flag = False
