from src.my_calculator import record_to_array_eval
import numpy as np
import pandas as pd

""" 该程序主要目的是：分析优化工具跟自己的优化算法的优化效果对比，即优化目标的函数值 """


def get_rec_dict(method, date):
    """
    读取txt文本信息并解析
    :param method: 优化方法
    :param date: 日期
    :return: 一个解析完的字典{指数：{日期：{key-value}}}
    """
    path = '../res/opt_weight/' + 'my_po-' + method + '-' + date + '.txt'

    with open(path, 'r') as file:
        lines = file.readlines()

    records = {'HS300': {}, 'ZZ500': {}, 'SZ50': {}}
    for line in lines:
        record_one_day = eval(line)
        benchmark = list(record_one_day.keys())[0]
        date_record = list(record_one_day[benchmark].keys())[0]
        record = eval(record_one_day[benchmark][date_record])
        record_array = record_to_array_eval(record)
        records[benchmark] = {**records[benchmark], **{date_record: record_array}}

    return records


def get_value_neu(records):
    """
    风险中性的投资组合目标函数为 
    min -wr + c||w-w_t||_1 + l/2||wX - w_bX||^2
            s.t.    0 <= w(i) <= upper
                    sum(w(i)) = 1
    :param records:记录字典 
    :return: （目标函数值，是否满足第一个约束，是否满足第二个约束）
    """
    w = records['w']
    r = records['r']
    c = records['c']
    w_t = records['w_o']
    l = records['l']
    X = records['X']
    w_b = records['w_b']
    upper = records['upper']

    result = -np.dot(w, r) + c * np.linalg.norm(w - w_t, 1) + l / 2 * np.linalg.norm((w - w_b) @ X, 2) ** 2
    ineq_con = (0 <= w) & (w <= upper)
    eq_con = np.sum(w) == 1
    return result, ineq_con, eq_con


def get_value_expo(records):
    """
    适度风险暴露的投资组合目标函数为
    min -wr + c||w-w_t||_1
        s.t.   0 <= w(i) <= upper
               sum(w(i)) = 1 
               expo_ <= (w - w_b)X <= expo+
    :param records: 记录字典
    :return: （目标函数值，是否满足第一个约束，是否满足第二个约束）
    """
    w = records['w']
    r = records['r']
    c = records['c']
    w_t = records['w_o']
    X = records['X']
    w_b = records['w_b']
    upper = records['upper']
    exposure = records["exposure"]

    result = -np.dot(w, r) + c * np.linalg.norm(w - w_t, 1)
    ineq_con = (0 <= w) & (w <= upper)
    eq_con = np.sum(w) == 1
    ineq_con_expo = (-exposure <= (w - w_b) @ X) & ((w - w_b) @ X <= exposure)
    return result, ineq_con, eq_con, ineq_con_expo


def compare_neu(records1, records2):
    """
    该函数用于同一个参考基准中比较两个风险中性的投资组合优化权重结果。
    返回的第一个值衡量标准是：目标函数小的次数多的方法
    :param records1: 优化方法1得到的记录字典
    :param records2: 优化方法2得到的记录字典
    :return: 效果较好的方法（record1或者record2），明细数据{'record1':{日期: 优化方法1对应的目标函数值和是否满足约束条件}，'record2':同1}
    """
    from collections import Counter
    res_date = {}
    detail_date = {'record1': {}, 'record2': {}}
    for date in records1.keys():
        rec1 = records1[date]
        rec2 = records2[date]
        val1, ineq_con1, eq_con1 = get_value_neu(rec1)
        val2, ineq_con2, eq_con2 = get_value_neu(rec2)
        res_date[date] = 'record1 is better' if val1 < val2 else ('record2 is better' if val1 > val2 else 'Same')
        detail_date['record1'][date] = (val1, ineq_con1, eq_con1)
        detail_date['record2'][date] = (val2, ineq_con2, eq_con2)
    stat_cnt = Counter(list(res_date.values()))
    better_record = 'Same'
    if 'record1 is better' in stat_cnt.keys() or 'record2 is better' in stat_cnt.keys():
        if stat_cnt['record1 is better'] > stat_cnt['record2 is better']:
            better_record = 'record1 is better'
        else:
            better_record = 'record2 is better'
    return better_record, detail_date


def compare_exposure(records1, records2):
    """
    该函数用于同一个参考基准中比较两个适度风险暴露的投资组合优化权重结果。
    返回的第一个值衡量标准是：目标函数小的次数多的方法
    :param records1: 优化方法1得到的记录字典
    :param records2: 优化方法2得到的记录字典
    :return: 效果较好的方法（record1或者record2），明细数据{'record1':{日期: 优化方法1对应的目标函数值和是否满足约束条件}，'record2':同1}
    """
    from collections import Counter
    res_date = {}
    detail_date = {'record1': {}, 'record2': {}}
    for date in records1.keys():
        rec1 = records1[date]
        rec2 = records2[date]
        val1, ineq_con1, eq_con1, ineq_con_expo1 = get_value_expo(rec1)
        val2, ineq_con2, eq_con2, ineq_con_expo2 = get_value_expo(rec2)
        res_date[date] = 'record1 is better' if val1 < val2 else ('record2 is better' if val1 > val2 else 'Same')
        detail_date['record1'][date] = (val1, ineq_con1, eq_con1, ineq_con_expo1)
        detail_date['record2'][date] = (val2, ineq_con2, eq_con2, ineq_con_expo2)
    stat_cnt = Counter(list(res_date.values()))
    better_record = 'Same'
    if 'record1 is better' in stat_cnt.keys() or 'record2 is better' in stat_cnt.keys():
        if stat_cnt['record1 is better'] > stat_cnt['record2 is better']:
            better_record = 'record1 is better'
        else:
            better_record = 'record2 is better'
    return better_record, detail_date


def compare(records1, records2, model='neu'):
    """
    该函数用于比较两个适度风险暴露的投资组合优化权重结果
    :param records1: 直接由txt文件生成的多个参考基准的优化参数权重字典1
    :param records2: 直接由txt文件生成的多个参考基准的优化参数权重字典2
    :return: 基准优化效果字典，如{'HS300': 'method1', 'ZZ500':'method2', 'SZ50': 'Same}
    """
    res = {}
    for benchmark in records1.keys():
        if model == 'neu':
            br, dd = compare_neu(records1[benchmark], records2[benchmark])
        else:
            br, dd = compare_exposure(records1[benchmark], records2[benchmark])
        res[benchmark] = br
    return res


def compare_with_cvxpy():
    """
    该函数为了对比自己的优化方法和cvxpy工具包的优化结果优劣性
    :return: 无。直接将结果集写入文件中
    """
    from copy import deepcopy
    from src.my_calculator import OptTool
    opt_tool = OptTool()

    fields_neu = ['result', 'ineq_con', 'eq_con', 'w', 'result_cvx', 'ineq_con_cvx', 'eq_con_cvx', 'w_cvx']
    fields_expo = ['result', 'ineq_con', 'eq_con', 'ineq_con_expo', 'w',
                   'result_cvx', 'ineq_con_cvx', 'eq_con_cvx', 'ineq_con_expo_cvx', 'w_cvx']

    # 获取记录字典
    method_neu = 'calc_weight'
    method_expo = 'calc_weight_with_exposure'
    the_date = '20200204'   # 算法时间
    record_neu = get_rec_dict(method_neu, the_date)
    record_expo = get_rec_dict(method_expo, the_date)

    for idx, opt_dict in record_neu.items():
        res_neu = {}
        for date, config in opt_dict.items():
            # 计算优化工具的优化权重
            w_tool = opt_tool.cvxpy_neu(config)
            config_cvxpy = deepcopy(config)
            config_cvxpy['w'] = w_tool

            res_neu[date] = list(get_value_neu(config)) + [config['w']] + list(get_value_neu(config_cvxpy)) + [w_tool]
        pd.DataFrame(res_neu, index=fields_neu).transpose().to_csv(f'../res/opt_weight_analysis/neu_analysis_{idx}.csv')

    for idx, opt_dict in record_expo.items():
        res_expo = {}
        for date, config in opt_dict.items():
            # 计算优化工具的优化权重
            w_tool = opt_tool.cvxpy_with_exposure(config)
            config_cvxpy = deepcopy(config)
            config_cvxpy['w'] = w_tool
            res_expo[date] = list(get_value_expo(config)) + [config['w']] + list(get_value_expo(config_cvxpy)) + [w_tool]
        pd.DataFrame(res_expo, index=fields_expo).transpose()\
            .to_csv(f'../res/opt_weight_analysis/expo_analysis_{idx}.csv')


if __name__ == '__main__':
    compare_with_cvxpy()
