import pandas as pd

""" 该程序的主要目的：统计每年，每17周的收益率变化情况 """


def change_codes_to_short(codes):
    codes = [c.replace('XSHE', 'SZ') for c in codes]
    codes = [c.replace('XSHG', 'SH') for c in codes]
    return codes


def stat_ret_each_year():
    df_ret = pd.read_csv('../data/ret/ret_weekly.csv', index_col=0)

    # 4年区间，每年51个周五，共204行记录
    res_dict = {}
    for i in range(4):
        res_dict[2016 + i] = df_ret.iloc[i * 51: i * 51 + 51].mean()
    df = pd.DataFrame(res_dict).transpose()
    df.loc['4年平均收益率'] = df.mean()
    df.loc['4年平均收益率/收益率标准差'] = df.mean() / df.std()
    # df.to_csv('../data/ret/ret_avg_year.csv')
    return df


def stat_ret_customize():
    df_ret = pd.read_csv('../data/ret/ret_weekly.csv', index_col=0)

    # 分成12个区间，每年51个周五，每17个周五为一个区间，共204行记录
    res_dict = {}
    for i in range(12):
        res_dict[i] = df_ret.iloc[i * 17: i * 17 + 17].mean()
    df = pd.DataFrame(res_dict).transpose()
    df.loc['平均收益率'] = df.mean()
    df.loc['平均收益率/收益率标准差'] = df.mean() / df.std()
    df.to_csv('../data/ret/ret_17week.csv')
    # return df


if __name__ == '__main__':
    df = pd.read_csv('../data/ret/ret_weekly.csv', index_col=0)
    df.columns = change_codes_to_short(df.columns)

    # df_ret = stat_ret_each_year()
    # df_ret.columns = change_codes_to_short(df_ret.columns)

    stat_ret_customize()

    print('test')
