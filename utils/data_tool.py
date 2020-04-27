import pandas as pd
import os


def get_index_composite():
    """
    该函数通过结合三个指数，生成三个指数的复合指数权重
    :return: 无，直接将结果写入data/index_weight/COM中
    """
    test_dates = pd.read_csv('../data/dates/test_date.csv', index_col=0)['date']
    test_dates = test_dates[test_dates >= '2016-02-01'].sort_values().tolist()
    for td in test_dates:
        df_res = pd.DataFrame()
        for ind in ["HS300", "SZ50", "ZZ500"]:
            # 取小于等于date_data的最大日期文件
            index_file_csv = sorted([p for p in os.listdir('../data/index_weight/%s' % ind)
                                     if p.split('.')[0] <= td]
                                    )[-1]
            file = '../data/index_weight/%s/%s' % (ind, index_file_csv)
            df_ind = pd.read_csv(file)
            df_res = pd.concat([df_res, df_ind])

        df_res = df_res.groupby(by='con_code').sum()
        df_res['weight'] = df_res['weight'] / 3
        df_res.to_csv('../data/index_weight/COMPO/%s.csv' % td)
        print(f"Total\t{len(test_dates)}\tFinish\t{test_dates.index(td)}\t{td}\t{df_res['weight'].sum()}")



if __name__ == "__main__":
    get_index_composite()
