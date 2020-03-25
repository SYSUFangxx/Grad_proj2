from rqalpha.api import *
import pandas as pd
import numpy as np

from src.refpo_calculator import ReferPO
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
    context.close_df = pd.read_csv('../data/allA_data/allAclose.csv', index_col=0)
    scheduler.run_weekly(trade, tradingday=1)


def trade(context, bar_dict):
    date_data = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
    print("Get data date:", date_data)

    ''' 获取仓位股票，用于生成总股池和前持仓权重，即all_stocks和list_pos_w '''
    position_stocks = list(context.portfolio.positions.keys())

    ''' 获取基准成分股，用于生成总股池和基准权重，即all_stocks和list_ben_w '''
    file = data_root + 'index_weight/%s/%s.csv' % (context.bench_root, date_data)
    df_ben = pd.read_csv(file, encoding='gbk', usecols=['code', 'i_weight'])
    ben_stocks = df_ben['code'].tolist()

    all_stocks = ben_stocks + [s for s in position_stocks if s not in ben_stocks]

    p = context.close_df.loc[:date_data, all_stocks].iloc[-5:]
    x_t = p.iloc[-1] / p.iloc[-2]
    x_t = x_t.dropna()
    x_t[moneyFund] = 1.00012
    x_t.index = change_code_format_to_long(x_t.index)

    '''采用my_po的good_codes'''
    # good_codes = GOOD_CODES
    # good_codes = change_code_format_to_long(good_codes)
    # pred_fix = pd.Series([0]*x_t.size, index=x_t.index)
    # pred_fix = pred_fix.apply(lambda x: 1.01 if x in good_codes else 0.99)
    # x_t = pred_fix
    '''采用my_po的good_codes'''

    drop_stocks = []
    for s in exclude_stocks:
        if s in x_t.index:
            drop_stocks.append(s)
    x_t = x_t.drop(drop_stocks)

    w_o = []
    po_stocks = list(x_t.index)
    for s in po_stocks:
        if s in position_stocks:
            w_o.append(context.portfolio.positions[s].value_percent)
        else:
            w_o.append(0)
    w_o = np.array(w_o)
    # if np.sum(w_o) != 0:
    #     w_o = w_o / np.sum(w_o)

    print("wosum", np.sum(w_o))

    po_inputs = {'w_o': w_o, 'x_t': x_t.values}

    # print("Nonzero w_o", [w for w in w_o if w])
    # print("x_t", x_t.values)

    if context.init_flag:
        weights = np.ones(len(po_stocks)) / len(po_stocks)
        context.init_flag = False
    else:
        weights = ReferPO.eg(po_inputs)

    print("Nonzero weights", [w for w in weights if w])

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
                    print("-" * 20, the_ord)
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
                    print("+" * 20, the_ord)
            else:
                w_adjust = weights[i] - w_o[i]
                quantity_adjust = \
                    context.stock_account.total_value * w_adjust / context.stock_account.positions[s].last_price
                if quantity_adjust < 100:
                    cnt_not_enough_share += 1
                elif quantity_adjust * bar_dict[s].close > context.stock_account.cash:
                    print('+' * 20, '不够资金', s, quantity_adjust, bar_dict[s].close, context.stock_account.cash)
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

    # # 多余的钱，全部买货币基金
    # print("remain_weight:", remain_weight)
    # if context.stock_account.cash > 0:
    #     print('$' * 10, '购买货币基金', '$' * 10)
    #     print("下单", moneyFund, context.stock_account.cash, "原有权重", w_o[po_stocks.index(moneyFund)],
    #           "原有市值", context.stock_account.positions[moneyFund].market_value)
    #     order_target_value(moneyFund, context.stock_account.cash)

