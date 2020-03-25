from rqalpha.api import *
import pandas as pd
import numpy as np
import os

strategy_abs_path = "../data/"


def init(context):
    #    ndays=len(tdaysData)

    context.counter = 0
    scheduler.run_daily(trade)
    context.root = strategy_abs_path + '000905-weights'


def trade(context, bar_dict):
    file = strategy_abs_path + 'dates/test_date.csv'
    df = pd.DataFrame()
    try:
        df = pd.read_csv(file, encoding='gbk')
    except IOError as e:
        print("Error: ", e)

    # dates=df.loc[559:,'date'].values # since 2016, 2016-01-08 to 2017-11-24
    dates = df.loc[609:, 'date'].values  # since 2017, 2017-01-06 to 2017-11-24

    tdaysData = list(dates.T)

    s = tdaysData[context.counter]
    my_date = '%s%s%s' % (s[0:4], s[5:7], s[8:10])
    cur_date = get_previous_trading_date(context.now).strftime('%Y%m%d')

    print("my_date, cur_date: "'%s %s' % (my_date, cur_date))

    if cur_date != my_date:
        pass
    else:
        context.counter = context.counter + 1
        date = get_previous_trading_date(context.now).strftime('%Y%m%d')
        date_save = get_previous_trading_date(context.now).strftime('%Y-%m-%d')
        # date = context.now.strftime('%Y%m%d')
        # date_save = context.now.strftime('%Y-%m-%d')
        print("date: ", date)

        file = strategy_abs_path + 'index_weight/HS300/%s.csv' % date_save
        df_hs300 = pd.DataFrame()
        try:
            df_hs300 = pd.read_csv(file, encoding='gbk')
        except IOError as e:
            print("Error: ", e)

        df_hs300.dropna(how='any', inplace=True)
        df_hs300 = df_hs300.reset_index(drop=True)

        hs300_stocks = df_hs300['code'].values
        hs300_weights = df_hs300['i_weight'].values

        pr_codes = ['600519.SH', '000858.SZ', '002304.SZ', '002415.SZ', '002236.SZ',
                    '600597.SH', '603288.SH', '000895.SZ', '601318.SH', '601336.SH',
                    '000651.SZ', '000333.SZ', '600036.SH', '600104.SH', '600816.SH',
                    '600674.SH', '600690.SH', '002081.SZ', '600886.SH', '000625.SZ',
                    '600887.SH', '000423.SZ', '600276.SH', '000963.SZ', '600535.SH',
                    '600271.SH', '002294.SZ', '600660.SH', '300072.SZ', '000538.SZ',
                    '600066.SH', '000002.SZ', '002271.SZ', '600406.SH', '600900.SH',
                    '002027.SZ', '000063.SZ', '002594.SZ', '601238.SH', '600518.SH',
                    '600703.SH', '600309.SH', '002310.SZ', '002466.SZ', '002142.SZ',
                    '601111.SH', '601211.SH', '600487.SH', '002008.SZ', '600383.SH'
            , '601288.SH', '601988.SH', '601398.SH'
                    ]
        pr_pr = np.ones(len(pr_codes)) * 0.01

        pr_codes = change_code_format(pr_codes)
        # tmpprcodes=remove_suspended_and_st(pr_codes)
        pr_codes_top = pr_codes
        print("len(pr_codes_top)", len(pr_codes_top))

        dict_pr = {}
        for i in range(0, len(pr_codes)):
            dict_pr[pr_codes[i]] = pr_pr[i]

        hs300_stocks = change_code_format(hs300_stocks)

        # print(c500_codes[0:10])

        dict_hs300 = {}
        for i in range(0, len(hs300_stocks)):
            dict_hs300[hs300_stocks[i]] = hs300_weights[i] / 100

        position_stocks = list(context.portfolio.positions.keys())
        all_stocks = set(pr_codes_top)
        all_stocks = all_stocks.union(hs300_stocks)
        all_stocks = all_stocks.union(position_stocks)

        file = strategy_abs_path + 'lncap_and_industry/%s.csv' % (date[:4]+'-'+date[4:6]+'-'+date[6:8])
        df_X = pd.DataFrame()
        try:
            df_X = pd.read_csv(file, encoding='gbk')
        except IOError as e:
            print("Error: ", e)
        df_X.index = df_X.code

        tmp_list = change_code_format(list(df_X.index.values))

        all_stocks = all_stocks.intersection(set(tmp_list))

        all_stocks = list(all_stocks)

        for iss in hs300_stocks:
            if iss not in tmp_list:
                print('missing hs300:%s' % iss)

        # print(all_stocks[0:10])


        n_stocks = len(all_stocks)
        print("n_stocks", n_stocks)
        list_pos_w = np.zeros(n_stocks)
        list_ben_w = np.zeros(n_stocks)
        list_pr = np.ones(n_stocks) * (-0.1)

        """
        wbsum: 基准权重之和
        wbmax：基准权重最大值
        wosum：原来的基准之和
        """
        wbsum = 0
        wbmax = 0
        wosum = 0
        for i in range(0, n_stocks):
            if all_stocks[i] in position_stocks:
                pos = context.portfolio.positions[all_stocks[i]]
                list_pos_w[i] = pos.value_percent
                wosum = wosum + list_pos_w[i]
            if all_stocks[i] in hs300_stocks:
                list_ben_w[i] = dict_hs300[all_stocks[i]]
                if list_ben_w[i] > wbmax:
                    wbmax = list_ben_w[i]
                wbsum = wbsum + list_ben_w[i]
            if all_stocks[i] in pr_codes:
                list_pr[i] = dict_pr[all_stocks[i]]
            else:
                pass
                # print('no pr %s' % all_stocks[i])

        print("wosum", wosum)
        print("wbsum", wbsum)
        print("wbmax", wbmax)
        for i in range(0, n_stocks):
            if wosum != 0:
                list_pos_w[i] = list_pos_w[i] / wosum
            if wbsum != 0:
                list_ben_w[i] = list_ben_w[i] / wbsum

                #       for i in range(0,20):
                #           print('%s: %f %f %f' %(all_stocks[i],list_pos_w[i],list_ben_w[i],list_pr[i]))

        industries = ['医药生物', '综合', '化工', '建筑材料', '有色金属', '采掘', '钢铁', '电气设备', '轻工制造', '食品饮料', '公用事业', '银行', '交通运输',
                      '休闲服务', '家用电器', '纺织服装', '建筑装饰', '计算机', '国防军工', '房地产', '传媒', '汽车', '机械设备', '非银金融', '通信', '商业贸易',
                      '电子', '农林牧渔']
        # fields=['LNCAP','ETOP', 'BTOP', 'CTOP','BETA','CMRA', 'RSTR', 'DASTD', 'HSIGMA','STOM', 'STOA', 'STOQ']

        fields = ['lncap']
        # fields=[]
        for ins in range(0, len(industries)):
            fields.append(industries[ins])
        all_ss = change_code_format_short(all_stocks)
        tmpX = df_X.loc[all_ss, fields].values

        # clusters=df.loc[list(all_stocks),['c0','c1','c2']].values
        # tmpX=np.concatenate((tmpX_org,clusters),axis=1)

        # tmp_cap对应着log市值
        tmp_cap = tmpX[:, 0].T
        n_cap = len(tmp_cap)
        for kkk in range(0, n_cap):
            tmpX[kkk, 0] = tmpX[kkk, 0] * tmpX[kkk, 0] * tmpX[kkk, 0]

        for p in range(0, len(fields)):
            temp = tmpX[:, p].T
            mean_ = temp.mean()
            std_ = temp.std()
            nn = len(temp)
            for q in range(0, nn):
                if std_ == 0:
                    tmpX[q, p] = 0
                else:
                    temp2 = (tmpX[q, p] - mean_) / std_
                    if temp2 > 3:
                        temp2 = 3
                    if temp2 < -3:
                        temp2 = -3
                    tmpX[q, p] = temp2


                    #        for p in range(0,len(fields)):
                    #            temp=tmpX[:,p].T
                    #            mean_=temp.mean()
                    #            nn=len(temp)
                    #            for q in range(0,nn):
                    #                if mean_ != 0:
                    #                    tmpX[q,p]=tmpX[q,p]/(mean_*nn)

        print("tmpX.shape", tmpX.shape)
        penalty = np.linalg.norm(np.dot(list_ben_w, tmpX)) ** 2
        print("penalty", penalty)
        penalty_r = np.linalg.norm(list_pr)
        print("penalty_r", penalty_r)
        weights = calc_weight(list_pos_w, list_ben_w, list_pr, tmpX, upper=0.04, c=0.006, l=0)
        # weights=calc_weight_with_exprosure(list_pos_w,list_ben_w,list_pr,tmpX,upper=0.04,c=0.006,exprosure=0.1)

        list_df = pd.DataFrame({'code': all_stocks, 'pos_w': list_pos_w, 'ben_w': list_ben_w, 'pr': list_pr})
        list_df.sort_values(by='code', inplace=True)

        test_df = pd.DataFrame(tmpX, index=all_stocks, columns=fields)
        test_df['code'] = all_stocks
        test_df['weight'] = weights
        test_df.sort_index(inplace=True)
        test_df = test_df[test_df['weight'] != 0]
        test_df = test_df[['code', 'weight'] + fields]

        tmp_sum = 0
        for i in range(0, len(weights)):
            tmp_sum = tmp_sum + weights[i]
        print('weights sum: %f' % tmp_sum)

        # buy_list 用于保存买卖股票的列表
        # trade_flags 用于保存是否需要修改仓位的标识
        buy_list = []
        trade_flags = []

        '''
        minw=0.000001

        for i in range(0,len(all_stocks)):
            if weights[i]<=minw:
                weights[i]=0

        for i in range(0,len(all_stocks)):
            if weights[i]>0:
                # 原先仓位为0
                # 或者 新仓位变动至少为1/4
                if list_pos_w[i]==0 or abs(weights[i]-list_pos_w[i])/list_pos_w[i]>=0.25:
                    buy_list.append(all_stocks[i])
                    trade_flags.append(True)
                else:
                    buy_list.append(all_stocks[i])
                    trade_flags.append(False)
            else:
                trade_flags.append(False)
        print('buy stocks %d' % len(buy_list))
'''

        for i in range(len(all_stocks)):
            if weights[i] > 0:
                buy_list.append(all_stocks[i])
                trade_flags.append(True)

                #        adjust_sum=0
                #        unchange_sum=0
                #        for i in range(0,len(all_stocks)):
                #            if weights[i]>0:
                #                if list_pos_w[i]>0 and abs(weights[i]-list_pos_w[i])/list_pos_w[i]<0.2:
                #                    adjust_sum=adjust_sum+weights[i]-list_pos_w[i]
                #                else:
                #                    unchange_sum=unchange_sum+weights[i]

                #        if unchange_sum !=0:
                #            for i in range(0,len(all_stocks)):
                #                if weights[i]>0:
                #                    if list_pos_w[i]>0 and abs(weights[i]-list_pos_w[i])/list_pos_w[i]<0.2:
                #                        weights[i]=list_pos_w[i]
                #                    else:
                #                        weights[i]=weights[i]*(1+adjust_sum/unchange_sum)

                #        check_sum=0
                #        for i in range(0,len(all_stocks)):
                #            check_sum=check_sum+weights[i]

                #        print('the check sum is %F' % check_sum)

        moneyFund = '510880.XSHG'

        for s in position_stocks:
            if s not in buy_list and s != moneyFund:
                # print('sell %s' % s)
                order_target_percent(s, 0)

        remain_weight = 1
        '''
        for i in range(0,len(all_stocks)):
            if weights[i]>0 and trade_flags[i]==True:
                #print('buy %s %f' %(all_stocks[i],weights[i]))
                order_target_percent(all_stocks[i], weights[i])
                remain_weight=remain_weight-weights[i]
            elif weights[i]>0 and list_pos_w[i]>0 and abs(weights[i]-list_pos_w[i])/list_pos_w[i]<0.25:
                remain_weight=remain_weight-list_pos_w[i]
'''
        for i in range(len(all_stocks)):
            if weights[i] > 0:
                order_target_percent(all_stocks[i], weights[i])
                remain_weight -= weights[i]

        # 多余的钱，全部买货币基金
        if remain_weight > 0:
            order_target_percent(moneyFund, remain_weight)

        my_list = {}
        my_list_weights = []
        for i in range(0, len(all_stocks)):
            if weights[i] > 0:
                my_list[change_format(all_stocks[i])] = weights[i]
                my_list_weights.append(weights[i] * 100)

        if remain_weight > 0:
            my_list[moneyFund] = remain_weight
            my_list_weights.append(remain_weight * 100)

        df_rt = pd.DataFrame(my_list_weights, index=my_list.keys(), columns=['weight'])
        df_rt.index.name = 'stock'

        file = strategy_abs_path + 'test_cal_stock_weight/%s.xls' % date_save

        # writer=pd.ExcelWriter(file)
        df_rt.to_excel(file)


# for ii in range(0,len_buy_list):  # 取因子值前10调仓买入
#            if lncap[ii] < split:
#                order_target_percent(buy_list[ii], s_ratio/ns)
#            else:
#            	order_target_percent(buy_list[ii], b_ratio/nb)


#        code_SHSZ=[]
#        for s in range(0,len(buy_list)):
#            tmp=buy_list[s].replace('XSHG','SH')
#            tmp=tmp.replace('XSHE','SZ')
#            code_SHSZ.append(tmp)


#        dfs=pd.DataFrame(np.ones(len(code_SHSZ))*100/len(code_SHSZ),index=code_SHSZ)
#        dfs.columns=['weight']
#        dfs.index.name='stock'

#        file=strategy_abs_path + 'abt_bt_data\\%s.csv' % date_save
#        dfs.to_csv(file)



def remove_suspended_and_st(stocks):
    rt = []
    n = len(stocks)
    for i in range(0, n):
        if is_suspended(stocks[i]) == False and is_st_stock(stocks[i]) == False:
            rt.append(stocks[i])
    return rt


def remove_st(stocks):
    rt = []
    n = len(stocks)
    for i in range(0, n):
        if is_st_stock(stocks[i]) == False:
            rt.append(stocks[i])
    return rt


def remove_small_lncap(stocks, lncaps):
    rt = []
    n = len(stocks)
    for i in range(0, n):
        # if lncaps[i]>23.4313 and lncaps[i]<24.63:#23.025851:24.63:
        if lncaps[i] > 23.719:  # 23.025851:# and lncaps[i]<23.719:
            # if lncaps[i]<24.1244632 and lncaps[i]>=23.4313:
            rt.append(stocks[i])
    return rt


def remove_low_ep(stocks, ep, lncaps):
    rt = []
    n = len(stocks)
    for i in range(0, n):
        if ep[i] > 0.02 and lncaps[i] > 23.719:
            rt.append(stocks[i])
    return rt


def change_code_format_long(stocks):
    rt = []
    for i in range(0, len(stocks)):
        tmp = stocks[i].replace('SH', 'XSHG')
        tmp = tmp.replace('SZ', 'XSHE')
        rt.append(tmp)
    return rt


def before_trading(context):
    # before_trading
    pass


def after_trading(context):
    stocks = context.portfolio.positions.keys()

    for i in range(0, len(stocks)):
        print('%s:%f' % (stocks[i], context.portfolio.positions[stocks[i]].value_percent))


# @assistant function
#  return the format of rqalpha
def change_code_format(stocks):
    if 'XSHG' in stocks[0] or 'XSHE' in stocks[0]:
        return stocks

    for i in range(0, len(stocks)):
        stocks[i] = stocks[i].replace('SH', 'XSHG')
        stocks[i] = stocks[i].replace('SZ', 'XSHE')
    return stocks


def change_code_format_short(stocks):
    rt = []
    for i in range(0, len(stocks)):
        tmp = stocks[i].replace('XSHG', 'SH')
        tmp = tmp.replace('XSHE', 'SZ')
        rt.append(tmp)
    return rt


def change_format(stock):
    stock = stock.replace('XSHG', 'SH')
    stock = stock.replace('XSHE', 'SZ')
    # print(stock)
    return stock


def simplex_projection(v):
    v_sorted = sorted(v, reverse=True)
    # print(v_sorted[0:3])
    n = len(v)
    max_ = 0
    for j in range(1, n):
        sum = 1
        for k in range(0, j):
            sum = sum - (v_sorted[k] - v_sorted[j])
        if sum > 0:
            max_ = j
    max_ = max_ + 1

    sigma = 0
    for i in range(0, max_):
        sigma = sigma + v_sorted[i]
    sigma = (sigma - 1) / max_

    w = np.zeros(len(v))
    for i in range(0, n):
        w[i] = max(0, v[i] - sigma)
    return w


def calc_weight(w_o, w_b, r, X, upper=0.02, c=0.006, l=100):
    # X: nstocks*nfactors
    # w_o: 1*nstocks
    # w_b: 1*nstocks
    # r: 1*nstocks

    y = np.dot(w_o - w_b, X)

    n = len(w_o)
    w = np.ones(n) / n
    v = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)
    q = np.zeros(n)
    z = np.ones(n) / n
    h = np.zeros(n)

    mu = 1e-4
    eps = 1e-8
    maxmu = 1e10
    rho = 1.02

    T = 0

    while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(u - v) > eps or np.linalg.norm(
                w - z) > eps) and T < 10000:

        # update v
        threshold = c / (2 * mu)
        pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

        for i in range(0, n):
            v[i] = min(0, pj_vec[i] + threshold) + max(0, pj_vec[i] - threshold)

        part1 = r - q + mu * v - l * np.dot(y, X.T)
        part2 = l * np.dot(X, X.T) + mu * np.identity(n)
        u = np.dot(part1, np.linalg.inv(part2))
        w = simplex_projection((v + w_o + z) / 2 + (h - p) / (2 * mu))
        for i in range(0, n):
            z[i] = max(0, min(w[i] - h[i] / mu, upper))

        p = p + mu * (w - v - w_o)
        q = q + mu * (u - v)
        h = h + mu * (z - w)
        mu = min(mu * rho, maxmu)
        T = T + 1
        if T % 200 == 0 or (np.linalg.norm(w - w_o - v) <= eps and np.linalg.norm(u - v) <= eps and np.linalg.norm(
                    w - z) <= eps):
            total = 0
            for kk in range(0, n):
                total = total + abs(w[kk] - w_o[kk])
            p3 = np.linalg.norm(np.dot(w - w_b, X))
            p2 = np.dot(w, r.T)
            loss = c * total - p2 + 0.5 * l * p3 * p3
            print('iter %d loss %f cost %f return %f risk %f' % (T, loss, total, p2, p3 * p3))

    return w


def calc_weight_with_exprosure(w_o, w_b, r, X, upper=0.02, c=0.006, exprosure=0.05):
    # X: nstocks*nfactors
    # w_o: 1*nstocks
    # w_b: 1*nstocks
    # r: 1*nstocks

    y = np.dot(w_o - w_b, X)

    n = len(w_o)
    m = X.shape[1]

    w = np.ones(n) / n
    v = np.zeros(n)
    u = np.zeros(n)
    p = np.zeros(n)
    q = np.zeros(n)
    z = np.ones(n) / n
    h = np.zeros(n)

    g = np.zeros(m)
    s = np.zeros(m)
    t_low = np.ones(m) * (-exprosure)
    t_upper = np.ones(m) * (exprosure)

    #    t_low[1]=0
    #    t_upper[1]=2*exprosure
    #    t_low[10]=0
    #    t_upper[10]=2*exprosure
    #    t_low[15]=0
    #    t_upper[15]=2*exprosure
    #    t_low[12]=0
    #    t_upper[12]=2*exprosure

    #    t_low[18]=-2*exprosure
    #    t_upper[18]=0
    #    t_low[19]=-2*exprosure
    #    t_upper[19]=0
    #    t_low[28]=-2*exprosure
    #    t_upper[28]=0



    mu = 1e-4
    eps = 1e-8
    maxmu = 1e10
    rho = 1.02

    T = 0

    while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(np.dot(u, X) + y - s) > eps or np.linalg.norm(
                u - v) > eps or np.linalg.norm(w - z) > eps) and T < 10000:

        # update v
        threshold = c / (2 * mu)
        pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

        for i in range(0, n):
            v[i] = min(0, pj_vec[i] + threshold) + max(0, pj_vec[i] - threshold)

        part1 = r - q + mu * v - mu * np.dot(y, X.T) + mu * np.dot(s, X.T) - np.dot(g, X.T)
        part2 = mu * np.dot(X, X.T) + mu * np.identity(n)
        u = np.dot(part1, np.linalg.inv(part2))

        w = simplex_projection((v + w_o + z) / 2 - (h + p) / (2 * mu))

        for i in range(0, n):
            z[i] = max(0, min(w[i] + h[i] / mu, upper))

        tmp_vec = np.dot(u, X) + y + g / mu
        for i in range(0, m):
            s[i] = max(t_low[i], min(tmp_vec[i], t_upper[i]))

        p = p + mu * (w - v - w_o)
        q = q + mu * (u - v)
        h = h + mu * (w - z)
        g = g + mu * (np.dot(u, X) + y - s)
        mu = min(mu * rho, maxmu)
        T = T + 1
        if T % 200 == 0 or (np.linalg.norm(np.dot(u, X) + y - s) <= eps and np.linalg.norm(
                        w - w_o - v) <= eps and np.linalg.norm(u - v) <= eps and np.linalg.norm(w - z) <= eps):
            total = 0
            for kk in range(0, n):
                total = total + abs(w[kk] - w_o[kk])
            p3 = np.linalg.norm(np.dot(w - w_b, X))
            p2 = np.dot(w, r.T)
            loss = c * total - p2
            print('iter %d loss %f cost %f return %f risk_expro %f' % (T, loss, total, p2, p3 * p3))

    return w

