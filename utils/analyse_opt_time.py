import pandas as pd

log_root = '../res/log_time/'
filter_words = ["Time Cost", "SZ50", "ZZ500", "HS300"]


def parse_key_line():
    for method in ["calc_weight", "calc_weight_with_exposure"]:
        with open(log_root + f"log_my_po-{method}20200412.log", 'r') as logfile:
            file = method + "_res.txt"
            res_file = open(log_root + file, 'w')

            for line in logfile.readlines():
                for fw in filter_words:
                    if fw in line:
                        res_file.write(line)
                        continue

            res_file.close()


def parse_time_cost():
    cols = ['total_time_cost', 'iter_times', 'avg_time_cost', 'method_benchmark']
    res_df = pd.DataFrame()
    for method in ["calc_weight", "calc_weight_with_exposure"]:
        file = open(log_root + method + '_res.txt', 'r')

        datas = []
        for line in file.readlines():
            if line.strip() in ["HS300", "ZZ500", "SZ50"]:
                df = pd.DataFrame(datas)
                df.loc[:, df.columns[-1] + 1] = method + '.' + line.strip()
                res_df = res_df.append(df)
                continue
            items = line.split('\t')
            total = eval(items[2])
            avg_time = eval(items[3])
            iter_times = int(total / avg_time)
            datas.append([total, iter_times, avg_time])

    res_df.columns = cols
    res_df = res_df[[cols[-1]] + cols[:3]]
    res_df.to_csv(log_root + 'time_cost_last_relative_price.csv', index=False)


if __name__ == '__main__':
    parse_key_line()
    parse_time_cost()
