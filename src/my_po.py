from rqalpha.api import *
import pandas as pd
import os

from src.my_calculator import MyCalculator, OptTool
from const.special_codes import *
from const.benchmark import *

# 常量
moneyFund = MONEY_FUND
exclude_stocks = EXCLUDE_STOCKS

data_root = "../data/"


def change_code_format_to_long(stocks):
    stocks = [s.replace('SH', 'XSHG') for s in stocks]
    stocks = [s.replace('SZ', 'XSHE') for s in stocks]
    return stocks


def get_ben_root(code):
    for k, v in BENCHMARK.items():
        if v == code:
            return k


def init(context):
    context.counter = 0
    # context.close_df = pd.read_csv('../data/allA_data/allAclose.csv', index_col=0)
    # context.close_df.columns = change_code_format_to_long(list(context.close_df.columns))
    # context.ret_df = context.close_df.pct_change().shift(-1)
    context.df_ret = pd.read_csv(data_root + '../data/ret/ret_weekly.csv', index_col=0)
    context.df_ret = context.df_ret.shift(-1)
    context.bench_root = get_ben_root(context.config.base.benchmark)
    scheduler.run_weekly(trade, tradingday=1)


def trade(context, bar_dict):
    date_data = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
    print("Get data date:", date_data)

    # 基准成分股，生成成分股权重s_ben_weights、基准成分股ben_stocks
    index_file_csv = sorted([p for p in os.listdir(data_root + 'index_weight/%s' % context.bench_root)
                             if p.split('.')[0] <= date_data]
                            )[-1]   # 取小于等于date_data的最大日期文件
    file = data_root + 'index_weight/%s/%s' % (context.bench_root, index_file_csv)
    df_ben = pd.read_csv(file, encoding='gbk', usecols=['con_code', 'weight'])
    df_ben.dropna(inplace=True)
    df_ben = df_ben.reset_index(drop=True)
    df_ben['con_code'] = change_code_format_to_long(df_ben['con_code'])
    df_ben['weight'] = df_ben['weight'] / 100
    ben_stocks = df_ben['con_code'].tolist()
    s_ben_weights = df_ben['weight'].copy()
    s_ben_weights.index = ben_stocks

    # 因子文件，这里只包含对数市值与行业风格因子
    # file = data_root + 'lncap_and_industry/%s.csv' % date_data
    file_mv = data_root + 'market_value/%s.csv' % date_data
    file_ind = data_root + 'industry/industry_sw1_one_hot.csv'
    df_mv = pd.read_csv(file_mv)
    df_ind = pd.read_csv(file_ind, encoding='gbk')

    # df_X = pd.read_csv(file, encoding='gbk')
    df_X = pd.merge(df_mv, df_ind, how='inner', left_on='ts_code', right_on='con_code')
    df_X.index = change_code_format_to_long(df_X.con_code)

    # 价值股票
    # good_stocks = change_code_format_to_long(list(set(GOOD_CODES)))
    good_stocks = change_code_format_to_long(list(set(GOOD_CODES_YEAR[date_data[:4]])))

    # 持仓股票
    position_stocks = list(context.portfolio.positions.keys())

    # 总股池股票，包含价值股票、基准成分股、持仓股票，且有因子数据、不在去除取票序列中的股票
    all_stocks = (set(good_stocks) | set(ben_stocks) | set(position_stocks)) & set(df_X.index) - set(exclude_stocks)
    all_stocks = list(all_stocks)

    # X_all -> 优化问题中的 X
    # 对对应的因子进行标准化，例如对数市值因子
    fields = ['ln_total_mv']
    for fd in fields:
        mean_fd = df_X[fd].mean()
        std_fd = df_X[fd].std()
        if std_fd == 0:
            df_X[fd] = 0
            continue
        df_X[fd] = df_X[fd].apply(lambda x: (x - mean_fd) / std_fd)
        # 异常值处理
        df_X[fd] = df_X[fd].apply(lambda x: 3 if x > 3 else (-3 if x < -3 else x))

    industries = ['医药生物', '综合', '化工', '建筑材料', '有色金属', '采掘', '钢铁', '电气设备', '轻工制造', '食品饮料',
                  '公用事业', '银行', '交通运输', '休闲服务', '家用电器', '纺织服装', '建筑装饰', '计算机', '国防军工',
                  '房地产', '传媒', '汽车', '机械设备', '非银金融', '通信', '商业贸易', '电子', '农林牧渔']
    fields += industries
    X_all = df_X.loc[all_stocks, fields].values

    # 已有仓位：pos_all -> 优化问题中的 w_0
    pos_hold = {k: v.value_percent for k, v in context.portfolio.positions.items()}
    s_pos = pd.Series(pos_hold)
    index_pos_stock = list(set(position_stocks) & set(df_X.index) - set(exclude_stocks))
    s_pos_all = pd.Series(0, index=all_stocks)
    s_pos_all[index_pos_stock] = s_pos[index_pos_stock]
    pos_all = s_pos_all[all_stocks].values

    # 参考基准的成分股权重：w_ben_all -> 优化问题中的 w_b
    s_w_ben = pd.Series(0, index=all_stocks)
    index_ben_stock = list(set(ben_stocks) & set(df_X.index) - set(exclude_stocks))
    s_w_ben[index_ben_stock] = s_ben_weights[index_ben_stock]
    w_ben_all = s_w_ben[all_stocks].values

    # 预测收益：ret_all -> 优化问题中的 r
    s_ret = pd.Series(-0.01, index=all_stocks)
    index_good_stock = list(set(good_stocks) & set(df_X.index) - set(exclude_stocks))
    s_ret[index_good_stock] = 0.01
    ret_all = s_ret[all_stocks].values

    # """ 使用周收益率作为预测 """
    # ret_all = context.df_ret[all_stocks].loc[date_data].fillna(0).values

    config = {
            "w_o": pos_all,
            "w_b": w_ben_all,
            "r": ret_all,
            "X": X_all
    }
    for k, v in context.CONFIG_CW.items():
        config[k] = v

    # 使用对应的算法优化投资组合权重
    if "calc_weight" in config['calc_method']:
        calc_func = getattr(MyCalculator, config['calc_method'])
    else:
        calc_func = getattr(OptTool, config['calc_method'])
    weights = calc_func(config)

    ''' 记录参数和权重，写入文件中 '''
    # record = record_to_str_eval({**config, **{'w': weights}})
    # context.opt_file.write(str({context.bench_record: {context.now.strftime('%Y%m%d'): record}}) + '\n')

    ''' 股票仓位调整 '''
    res_weights = pd.Series(weights, index=all_stocks)

    # 将没有新权重的持仓股票卖出
    for s in position_stocks:
        if s not in res_weights[res_weights > 0].index and s != moneyFund:
            order_target_percent(s, 0)

    remain_weight = 1
    for stk_code, stk_weight in res_weights.iteritems():
        if stk_weight > 0:
            the_ord = order_target_percent(stk_code, stk_weight)
            remain_weight -= stk_weight
            if the_ord:
                if the_ord.status == ORDER_STATUS.FILLED:
                    remain_weight - stk_weight
                else:
                    print(str(the_ord.get_state()), "*!*!*!*!*!*--订单失败--*!*!*!*!*!*")
            else:
                pass
