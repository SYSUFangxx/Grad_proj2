import tushare as ts
import pandas as pd


class TSData:
    def __init__(self):
        # print("该类的功能是拿各种tushare的数据")
        self.pro = ts.pro_api('c7fcf12b6643c186bee3411c7df9f14b24d57956d632898bdfdb6361')
        pass

    def get_all_a_stocks(self):
        df_stk = self.pro.stock_basic(exchange='', fields='ts_code,symbol,name,fullname,area,industry,exchange,list_date,delist_date')
        return df_stk.ts_code.tolist()

    def get_indexes_weight(self, start_date=20160101, end_date=20200101, res_root='../data/index_weight/'):
        """
        获取沪深300、上证50、中证500的指数成分股及其权重        
        :param start_date: 获取数据的初始时间
        :param end_date: 获取数据的结束时间
        :param res_root:结果数据集的文件夹位置  
        :return:无返回，直接将各指数成分权重写入res_root中，对应子文件夹为HS300/SZ50/ZZ500 
        """
        e_date = start_date

        # 因为tushare中沪深300是深交所代码，因此用“399300.SZ”的指数代码
        index_code = {'HS300': '399300.SZ', 'SZ50': '000016.SH', 'ZZ500': '000905.SH'}

        while True:
            s_date = e_date
            e_date = s_date // 10000 * 10000 + ((s_date // 100 % 100) + 3) // 12 * 10000 + (
                                                                            (s_date // 100 % 100) + 3) % 12 * 100 + 1
            if e_date > end_date:
                break

            print(f"\n获取数据，区间{s_date}~{e_date}")

            for idx, code in index_code.items():
                df_index = self.pro.index_weight(index_code=code, start_date=s_date, end_date=e_date)
                dates = set(df_index.trade_date)
                for d in dates:
                    df = df_index[df_index.trade_date == d][['con_code', 'weight']]
                    file_name = d[:4] + '-' + d[4:6] + '-' + d[6:] + '.csv'
                    df.to_csv(res_root + idx + '/' + file_name, index=False)
                    print(f"获取指数数据\t{code}\t{d}")

    def get_market_value(self, res_root = '../data/market_value/'):
        """
        获取市场上股票的市值信息
        :param res_root: 结果数据集的文件夹位置  
        :return: 无返回，直接将各指数成分权重写入res_root中
        """
        import numpy as np
        # 实验区间的交易日，list
        trd_dates = eval(open('../data/dates/trading_days.txt', 'r').readline())
        for d in trd_dates:
            df_mv = self.pro.daily_basic(ts_code='', trade_date=d.replace('-', ''), fields='ts_code,total_mv')
            df_mv['ln_total_mv'] = np.log(df_mv['total_mv'])
            df_mv.to_csv(res_root + d + '.csv', index=False)
            print(f"获取{d}的市值数据")

    def get_sw_industry(self, res_root='../data/industry/'):
        """
        获取申万行业分类
        :param res_root: 结果数据集的文件夹位置  
        :return: 无返回，直接将各指数成分权重写入res_root中
        """
        # 申万一级行业分类
        df_sw = self.pro.index_classify(level='L1', src='SW')

        # 股票信息，获取股票中文名称
        df_stk = self.pro.stock_basic(exchange='', fields='ts_code,name,list_date,delist_date')

        df_res = pd.DataFrame()
        for _, row in df_sw.iterrows():
            ind_code = row['index_code']
            ind_name = row['industry_name']
            df_ind = self.pro.index_member(index_code=ind_code, fields='index_code,con_code,in_date,out_date,is_new')
            df_res = pd.concat([df_res, df_ind])
            print(f"获取申万一级行业分类，{ind_code},{ind_name}，数据量为{df_ind.shape}")

        # 左关联，补充行业中文名
        df_res = pd.merge(left=df_res, right=df_sw, on='index_code', how='left')
        # 左关联，补充股票中文名
        df_res = pd.merge(left=df_res, right=df_stk, left_on='con_code', right_on='ts_code', how='left')

        df_res.to_csv(res_root + 'industry_sw1.csv', index=False)

if __name__ == '__main__':
    ts_data = TSData()

    # # 获取指数权重数据
    # ts_data.get_indexes_weight()

    # # 获取股票的市值信息
    # ts_data.get_market_value()

    # 获取申万一级行业分类
    ts_data.get_sw_industry()
