import os

from rqalpha.api import *
import pandas as pd
import numpy as np

from src.refpo_calculator import ReferPO

from const.special_codes import *
from const.benchmark import BENCHMARK

# 去除股票
exclude_stocks = EXCLUDE_STOCKS
# 货币基金
moneyFund = MONEY_FUND
# 存储数据路径
data_root = "../data/"


# @assistant function
#  return the format of rqalpha
def change_code_format_to_long(stocks):
    stocks = [s.replace('SH', 'XSHG') for s in stocks]
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
    context.close_df = pd.read_csv('../data/stocks/all_a_close_rqalpha.csv', index_col=0)
    context.close_df.columns = change_code_format_to_long(context.close_df.columns)
    scheduler.run_weekly(trade, tradingday=1)


def trade(context, bar_dict):
    date_data = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
    print("Get data date:", date_data)

    ''' 获取仓位股票，用于生成总股池和前持仓权重，即all_stocks和list_pos_w '''
    position_stocks = list(context.portfolio.positions.keys())

    ''' 获取基准成分股，用于生成总股池和基准权重，即all_stocks和list_ben_w '''
    index_file_csv = sorted([p for p in os.listdir(data_root + 'index_weight/%s' % context.bench_root)
                             if p.split('.')[0] <= date_data]
                            )[-1]  # 取小于等于date_data的最大日期文件
    file = data_root + 'index_weight/%s/%s' % (context.bench_root, index_file_csv)
    df_ben = pd.read_csv(file, encoding='gbk', usecols=['con_code', 'weight'])
    df_ben['con_code'] = change_code_format_to_long(df_ben['con_code'])
    ben_stocks = df_ben['con_code'].tolist()

    all_stocks = ben_stocks + [s for s in position_stocks if s not in ben_stocks]

    omega = 5

    """ 采用论文中的模型，p_max / p 作为相对价格的估计 """
    p = context.close_df.loc[:date_data, all_stocks].iloc[-omega:]
    x_pred = p.min() / p.iloc[-1]

    # """ 采用close计算未来收益 """
    # p = context.close_df.loc[date_data:, all_stocks].iloc[:omega]
    # x_pred = p.iloc[-1] / p.iloc[0]

    """采用good_codes计算未来收益"""
    good_codes = change_code_format_to_long(GOOD_CODES)
    x_pred[[s for s in x_pred.index if s in good_codes]] = 1.1
    # for code in good_codes:
    #     x_pred[code] = 1.1
    # x_pred[[s for s in x_pred.index if s not in good_codes]] = 0.99

    x_pred = x_pred.dropna()
    x_pred[moneyFund] = 1.00012

    drop_stocks = [s for s in x_pred.index if is_suspended(s) or is_st_stock(s) or s in exclude_stocks]
    x_pred = x_pred.drop(drop_stocks)

    w_o = []
    po_stocks = list(x_pred.index)
    for s in po_stocks:
        if s in position_stocks:
            w_o.append(context.portfolio.positions[s].value_percent)
        else:
            w_o.append(0)
    w_o = np.array(w_o)
    # if np.sum(w_o) != 0:
    #     w_o = w_o / np.sum(w_o)

    print("wosum", np.sum(w_o))

    po_inputs = {'w_o': w_o, 'x_pred': x_pred.values}

    """ 20200404 以下条件判断没有必要，第一次则w_o为0即可 """
    # if context.init_flag:
    #     weights = np.ones(len(po_stocks)) / len(po_stocks)
    #     context.init_flag = False
    # else:
    # weights = ReferPO.sspo(po_inputs)
    weights = ReferPO.sspo_cvxpy(po_inputs)

    non_zero_w = [(s, w) for s, w in zip(po_stocks, weights) if w]
    print("Nonzero weights", len(non_zero_w), non_zero_w)

    ''' 权重之和检查 '''
    check_sum = 0
    for iw in weights:
        check_sum += iw
    if check_sum != 1:
        print('weights sum: %f' % check_sum)

    ''' 股票仓位调整 '''
    # 股市做多约束，进行权重调整
    weights = adjust_weights_to_pos(weights)
    remain_weight = 1

    # 减仓调整
    cnt_not_enough_share = 0
    cnt_else = 0
    for i, s in enumerate(po_stocks):
        if w_o[i] > weights[i]:
            pos_weight = w_o[i]
            the_ord = order_target_percent(s, weights[i])
            if the_ord:
                if the_ord.status == ORDER_STATUS.FILLED:
                    pos_weight = weights[i]
                else:
                    # print("-" * 20, ord)
                    pass
            else:
                w_adjust = w_o[i] - weights[i]
                quantity_adjust = \
                    context.stock_account.total_value * w_adjust / context.stock_account.positions[s].last_price
                if quantity_adjust < 100:
                    cnt_not_enough_share += 1
                else:
                    cnt_else += 1
            remain_weight -= pos_weight
    if cnt_not_enough_share or cnt_else:
        print('-' * 50, "减仓配资过程中\n不够一手{}\t其他{}".format(cnt_not_enough_share, cnt_else))

    # 加仓调整
    cnt_not_enough_cash = 0
    cnt_not_enough_share = 0
    cnt_else = 0
    for i, s in enumerate(po_stocks):
        if s == moneyFund:
            continue
        if w_o[i] < weights[i]:
            pos_weight = w_o[i]
            the_ord = order_target_percent(s, weights[i])
            if the_ord:
                if the_ord.status == ORDER_STATUS.FILLED:
                    pos_weight = weights[i]
                else:
                    # print("+" * 20, ord)
                    pass
            else:
                w_adjust = weights[i] - w_o[i]
                quantity_adjust = \
                    context.stock_account.total_value * w_adjust / context.stock_account.positions[s].last_price
                if quantity_adjust < 100:
                    cnt_not_enough_share += 1
                elif quantity_adjust * bar_dict[s].close > context.stock_account.cash:
                    print('+' * 20, '不够资金', s, "股数", quantity_adjust, "股价", bar_dict[s].close,
                          "账户余额", context.stock_account.cash)
                    cnt_not_enough_cash += 1
                else:
                    print('+' * 20, '其他', s, "股数", quantity_adjust, "账户余额", context.stock_account.cash)
                    cnt_else += 1
                # print("：ord is NoneType")
                # print("总账户余额：{}，需买{}，{}股，最新价格{}，"
                #       "调整前：{}，调整后：{}，权重差值：{}".format(context.stock_account.cash, s, quantity_adjust,
                #                                      bar_dict[s].close, w_o[i], weights[i], w_adjust))
            remain_weight -= pos_weight

    if cnt_not_enough_cash or cnt_not_enough_share or cnt_else:
        print('+' * 50, "加仓配资过程中\n不够资金{}\t不够一手{}\t其他{}".format(cnt_not_enough_cash,
                                                               cnt_not_enough_share, cnt_else))

    # 多余的钱，全部买货币基金
    print("remain_weight:", remain_weight)
    if context.stock_account.cash > 0:
        print('$' * 10, '购买货币基金', '$' * 10)
        print("下单", moneyFund, context.stock_account.cash, "原有权重", w_o[po_stocks.index(moneyFund)],
              "原有市值", context.stock_account.positions[moneyFund].market_value)
        order_target_value(moneyFund, context.stock_account.cash)
