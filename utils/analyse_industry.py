import pandas as pd
from const.special_codes_indexes import *

""" 该程序的主要目的是：分析各个指数、结果集中的行业分布情况 """


def change_codes_to_short(codes):
    codes = [c.replace('XSHE', 'SZ') for c in codes]
    codes = [c.replace('XSHG', 'SH') for c in codes]
    return codes


def analyse_res_industry(res_pos_path):
    # 回测每日持仓情况
    df_res = pd.read_csv(res_pos_path, encoding='gbk')
    df_res.order_book_id = change_codes_to_short(df_res.order_book_id)
    df_res = df_res[['date', 'order_book_id']]
    # 行业信息
    df_ind = pd.read_csv('../data/industry/industry_sw1_one_hot.csv', encoding='gbk')

    df = pd.merge(df_res, df_ind, how='inner', left_on='order_book_id', right_on='con_code')
    industries = ['医药生物', '综合', '化工', '建筑材料', '有色金属', '采掘', '钢铁', '电气设备', '轻工制造', '食品饮料',
                  '公用事业', '银行', '交通运输', '休闲服务', '家用电器', '纺织服装', '建筑装饰', '计算机', '国防军工',
                  '房地产', '传媒', '汽车', '机械设备', '非银金融', '通信', '商业贸易', '电子', '农林牧渔']
    df = df[['date'] + industries]
    df = df.groupby('date').sum()
    res_file = '-'.join(res_pos_path.split('/my_po/stock_positions.csv')[0]
                        .split('../res/backtest/my_po-')[1].split('/')) + '-ind_res.csv'
    df.to_csv('../data/analysis/' + res_file)


def analyse_gc_industry():
    # 行业信息
    df_ind = pd.read_csv('../data/industry/industry_sw1_one_hot.csv', encoding='gbk', index_col=0)

    df_gc1 = df_ind.loc[GOOD_CODES_1]
    df_gc1.loc['sum'] = df_gc1.sum()
    df_gc1.to_csv('../data/analysis/gc1_ind.csv')

    df_gc2 = df_ind.loc[GOOD_CODES_2]
    df_gc2.loc['sum'] = df_gc2.sum()
    df_gc2.to_csv('../data/analysis/gc2_ind.csv')

    df_gc1 = df_ind.loc[GOOD_CODES_1]
    df_gc1.loc['sum'] = df_gc1.sum()
    df_gc1.to_csv('../data/analysis/gc1_ind.csv')

    df_gc3 = df_ind.loc[GOOD_CODES_3]
    df_gc3.loc['sum'] = df_gc3.sum()
    df_gc3.to_csv('../data/analysis/gc3_ind.csv')

    df_gc = df_ind.loc[GOOD_CODES]
    df_gc.loc['sum'] = df_gc.sum()
    df_gc.to_csv('../data/analysis/gc_ind.csv')


def analyse_index_industry(path_ind):
    res_path = '../data/analysis' + path_ind.split('index_weight')[1].split('/2019-01-31')[0] + '_indstry20190131.csv'
    df_index = pd.read_csv(path_ind)

    # 行业信息
    df_ind = pd.read_csv('../data/industry/industry_sw1_one_hot.csv', encoding='gbk')

    df = pd.merge(df_index, df_ind, left_on='con_code', right_on='con_code', how='inner')
    industries = ['医药生物', '综合', '化工', '建筑材料', '有色金属', '采掘', '钢铁', '电气设备', '轻工制造', '食品饮料',
                  '公用事业', '银行', '交通运输', '休闲服务', '家用电器', '纺织服装', '建筑装饰', '计算机', '国防军工',
                  '房地产', '传媒', '汽车', '机械设备', '非银金融', '通信', '商业贸易', '电子', '农林牧渔']
    df.index = df['con_code']
    df = df[industries]
    df.loc['sum'] = df.sum()
    df.to_csv(res_path)


if __name__ == '__main__':
    method = 'calc_weight'

    for ind in ['HS300', 'SZ50', 'ZZ500']:
        path = f'../res/backtest/my_po-{method}/{ind}/my_po/stock_positions.csv'
        analyse_res_industry(path)
        path_ind = f'../data/index_weight/{ind}/2019-01-31.csv'
        analyse_index_industry(path_ind)

    analyse_gc_industry()
