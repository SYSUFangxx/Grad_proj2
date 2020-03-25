import cvxpy as cp
import numpy as np
import time
import os


def check_root(root):
    if not os.path.exists(root):
        os.mkdir(root)


def record_to_str_eval(config):
    """
    将权重、变量等信息转换成eval函数可以识别的string，用于txt文件持久化
    """
    res = {}
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            res[k] = v.tolist()
        else:
            res[k] = v
    return str(res)


def record_to_array_eval(config):
    """
    将txt文件中权重、变量等信息通过eval函数识别成array对象
    """
    res = {}
    for k, v in config.items():
        if isinstance(v, list):
            res[k] = np.array(v)
        else:
            res[k] = v
    return res


def simplex_projection(v):
    """
    这里对应John Duchi 2008年的论文中的figure1算法，用于求解单纯形上的欧式投影问题
    注意：
        z=1
        rho与j跟原论文的下标可能有些小差异，因为程序中数组是从0开始计数的
    :param v: 约束目标中的向量v
    :return: 算法求解的向量w
    """
    z = 1.
    v_sorted = sorted(v, reverse=True)
    n = len(v)
    rho = 0
    for j in range(1, n):
        sum_mur = 0
        for r in range(j):
            sum_mur += v_sorted[r]
        if v_sorted[j - 1] - (sum_mur - z) / j > 0:
            rho = j
        else:
            if j == 1:
                print(list(v))
                print(v_sorted)
                print('*' * 50)
                return np.zeros(v.size)
            break

    sum_mui = 0
    for i in range(rho):
        sum_mui += v_sorted[i]
    theta = (sum_mui - z) / rho

    w = np.zeros(len(v))
    for i in range(n):
        w[i] = max(0.0, v[i] - theta)
    return w


class MyCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calc_weight_with_sparsity(config):
        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, l=1, lam=0.5

        w_o = config["w_o"]
        w_b = config["w_b"]
        r = config["r"]
        X = config["X"]
        upper = config["upper"]
        c = config["c"]
        l = config["l"]
        lam = config["lam"]

        # X: nstocks*nfactors
        # w_o: 1*nstocks
        # w_b: 1*nstocks
        # r: 1*nstocks

        Y = np.dot(w_o - w_b, X)

        n = len(w_o)
        w = np.ones(n) / n
        v = np.zeros(n)
        u = np.zeros(n)
        p = np.zeros(n)
        q = np.zeros(n)
        z = np.ones(n) / n
        s = np.ones(n) / n
        o = np.zeros(n)
        h = np.zeros(n)

        mu = 1e-4
        eps = 1e-8
        maxmu = 1e10
        rho = 1.02

        T = 0

        while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(u - v) > eps
               or np.linalg.norm(w - z) > eps or np.linalg.norm(s - z) > eps) and T < 10000:

            # update v
            threshold = c / (2 * mu)
            pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)
            for i in range(0, n):
                v[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

            part_r = r - q + mu * v - l * np.dot(Y, X.T)
            part_l = l * np.dot(X, X.T) + mu * np.identity(n)
            u = np.dot(np.linalg.inv(part_l), part_r)
            w = simplex_projection((v + w_o + z) / 2 + (h - p) / (2 * mu))
            for i in range(0, n):
                val = (w[i] + s[i]) / 2 + (o[i] - h[i]) / (2 * mu)
                z[i] = max(0.0, min(val, upper))

            threshold = lam / mu
            pj_vec = z - o / mu
            for i in range(0, n):
                s[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

            p = p + mu * (w - v - w_o)
            q = q + mu * (u - v)
            h = h + mu * (z - w)
            o = o + mu * (s - z)
            mu = min(mu * rho, maxmu)
            T = T + 1
            if T % 200 == 0 or (np.linalg.norm(w - w_o - v) <= eps and np.linalg.norm(u - v) <= eps
                                and np.linalg.norm(w - z) <= eps and np.linalg.norm(s - z) <= eps):
                total = 0
                for kk in range(0, n):
                    total = total + abs(w[kk] - w_o[kk])
                p3 = np.linalg.norm(np.dot(w - w_b, X))
                p2 = np.dot(w, r.T)
                loss = c * total - p2 + 0.5 * l * p3 * p3
                sparsity = np.linalg.norm(w, ord=1)
                print('iter {} loss {} cost {} return {} risk {} sparsity {}'.format(T, loss, total, p2, p3 * p3, sparsity))
            # print(T)

        return w

    @staticmethod
    def calc_weight_lowtr_constrains(config):
        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, l=100

        # X: nstocks*nfactors
        # w_o: 1*nstocks
        # w_b: 1*nstocks
        # r: 1*nstocks

        w_o = config["w_o"]
        w_b = config["w_b"]
        r = config["r"]
        X = config["X"]
        upper = config["upper"]
        c = config["c"]
        l = config["l"]
        lam = config["lam"]

        mu = 1e-4
        eps = 1e-8
        maxmu = 1e10
        rho = 1.02

        Y = np.dot(w_o - w_b, X)

        n = len(w_o)
        w = np.ones(n) / n
        v = np.zeros(n)
        u = np.zeros(n)
        p = np.zeros(n)
        q = np.zeros(n)
        z = np.ones(n) / n
        h = np.zeros(n)

        T = 0

        while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(u - v) > eps or np.linalg.norm(
                    w - z) > eps) and T < 10000:

            # update v
            threshold = c / (2 * mu)
            pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

            for i in range(0, n):
                v[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

            part_r = r - q + mu * v - l * np.dot(Y, X.T)
            part_l = l * np.dot(X, X.T) + mu * np.identity(n)
            u = np.dot(np.linalg.inv(part_l), part_r)
            # print("v", len([nw for nw in (v + w_o + z) / 2 + (h - p) / (2 * mu) if nw]), [nw for nw in (v + w_o + z) / 2 + (h - p) / (2 * mu) if nw])
            w = simplex_projection((mu * (v + w_o + z) + 2 * lam) / (2 * (lam + mu)))
            # print("w", len([nw for nw in w if nw]), [nw for nw in w if nw])
            for i in range(0, n):
                z[i] = max(0.0, min(w[i] - h[i] / mu, upper))

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
                turnover = np.linalg.norm(w - w_o)
                p3 = np.linalg.norm(np.dot(w - w_b, X))
                p2 = np.dot(w, r.T)
                loss = c * total - p2 + 0.5 * l * p3 * p3
                print('iter {} loss {} cost {} return {} risk {} turnover {}'.format(T, loss, total, p2, p3 * p3, turnover))
            # print(T)

        return w

    @staticmethod
    def calc_weight(config):
        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, l=100
        start = time.time()

        # X: nstocks*nfactors
        # w_o: 1*nstocks
        # w_b: 1*nstocks
        # r: 1*nstocks

        w_o = config["w_o"]
        w_b = config["w_b"]
        r = config["r"]
        X = config["X"]
        upper = config["upper"]
        c = config["c"]
        l = config["l"]

        mu = 1e-4
        eps = 1e-8
        maxmu = 1e10
        rho = 1.02

        Y = np.dot(w_o - w_b, X)

        n = len(w_o)
        w = np.ones(n) / n
        v = np.zeros(n)
        u = np.zeros(n)
        p = np.zeros(n)
        q = np.zeros(n)
        z = np.ones(n) / n
        h = np.zeros(n)

        T = 0

        while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(u - v) > eps or np.linalg.norm(
                    w - z) > eps) and T < 10000:

            # update v
            threshold = c / (2 * mu)
            pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

            for i in range(0, n):
                v[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

            part_r = r - q + mu * v - l * np.dot(Y, X.T)
            part_l = l * np.dot(X, X.T) + mu * np.identity(n)
            u = np.dot(np.linalg.inv(part_l), part_r)
            # print("v", len([nw for nw in (v + w_o + z) / 2 + (h - p) / (2 * mu) if nw]), [nw for nw in (v + w_o + z) / 2 + (h - p) / (2 * mu) if nw])
            w = simplex_projection((v + w_o + z) / 2 + (h - p) / (2 * mu))
            # print("w", len([nw for nw in w if nw]), [nw for nw in w if nw])
            for i in range(0, n):
                z[i] = max(0.0, min(w[i] - h[i] / mu, upper))

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
                print('iter {} loss {} turnover {} return {} risk {}'.format(T, loss, total, p2, p3 * p3))
            # print(T)

        print(f"Time Cost: calc_weight\t{start}\t{time.time() - start}"
              f"\t{((time.time() - start) / T if T != 0 else 9999)}")
        return w

    @staticmethod
    def calc_weight_with_exposure(config):
        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, exposure=0.05
        start = time.time()

        # X: nstocks*nfactors
        # w_o: 1*nstocks
        # w_b: 1*nstocks
        # r: 1*nstocks

        w_o = config["w_o"]
        w_b = config["w_b"]
        r = config["r"]
        X = config["X"]
        upper = config["upper"]
        c = config["c"]
        # l = config["l"]
        exposure = config["exposure"]

        Y = np.dot(w_o - w_b, X)

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
        y = np.zeros(m)

        # 原有的风险暴露，常值控制
        t_low = np.ones(m) * (-exposure)
        t_upper = np.ones(m) * (exposure)

        # # 因子择时，控制风险敞口
        # t_low = exposure['-']
        # t_upper = exposure['+']

        mu = 1e-4
        eps = 1e-8
        maxmu = 1e10
        rho = 1.02

        T = 0

        while (np.linalg.norm(w - w_o - v) > eps or np.linalg.norm(np.dot(u, X) + Y - y) > eps or np.linalg.norm(
                    u - v) > eps or np.linalg.norm(w - z) > eps) and T < 10000:

            # update v
            threshold = c / (2 * mu)
            pj_vec = (w - w_o + u) / 2 + (p + q) / (2 * mu)

            for i in range(0, n):
                v[i] = min(0.0, pj_vec[i] + threshold) + max(0.0, pj_vec[i] - threshold)

            part1 = r - q + mu * v - mu * np.dot(Y, X.T) + mu * np.dot(y, X.T) - np.dot(g, X.T)
            part2 = mu * np.dot(X, X.T) + mu * np.identity(n)
            # u = np.dot(np.linalg.inv(part2), part1)
            u = np.dot(part1, np.linalg.inv(part2))

            w = simplex_projection((v + w_o + z) / 2 - (h + p) / (2 * mu))
            # w = simplex_projection((v + w_o + z) / 2 + (h - p) / (2 * mu))

            for i in range(0, n):
                z[i] = max(0.0, min(w[i] + h[i] / mu, upper))

            tmp_vec = np.dot(u, X) + Y + g / mu
            for i in range(0, m):
                y[i] = max(t_low[i], min(tmp_vec[i], t_upper[i]))

            p = p + mu * (w - v - w_o)
            q = q + mu * (u - v)
            h = h + mu * (w - z)
            g = g + mu * (np.dot(u, X) + Y - y)
            mu = min(mu * rho, maxmu)
            T = T + 1
            if T % 200 == 0 or (np.linalg.norm(np.dot(u, X) + Y - y) <= eps and np.linalg.norm(
                            w - w_o - v) <= eps and np.linalg.norm(u - v) <= eps and np.linalg.norm(w - z) <= eps):
                total = 0
                for kk in range(0, n):
                    total = total + abs(w[kk] - w_o[kk])
                p3 = np.linalg.norm(np.dot(w - w_b, X))
                p2 = np.dot(w, r.T)
                loss = c * total - p2
                print('iter {} loss {} cost {} return {} risk_expro {}'.format(T, loss, total, p2, p3 * p3))

        print(f"Time Cost: calc_weight\t{start}\t{time.time() - start}\t{((time.time() - start) / T)}")
        return w


class OptTool:
    """
    该类为已有的优化工具：Scipy，cvxpy等
    """
    def __init__(self):
        pass

    @staticmethod
    def SciOpt_neu(config):
        from scipy.optimize import minimize

        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, l=100
        start = time.time()

        # X: nstocks*nfactors
        # w_o: 1*nstocks
        # w_b: 1*nstocks
        # r: 1*nstocks

        # 定义变量
        w_t = config["w_o"]
        r = config["r"]
        X = config["X"]
        w_b = config["w_b"]
        upper = config["upper"]
        l = config["l"]
        c = config["c"]
        m = len(w_t)  # 股票数

        # 初始化: 原始权重为1/m
        w_0 = np.array([1 / m] * m)

        obj_fun = lambda w: - np.dot(w, r) + c * np.linalg.norm(w - w_t, 1) + l / 2 * np.linalg.norm(np.dot(w - w_b, X)) ** 2

        bound = tuple(tuple(t) for t in [[0.0, upper]] * m)

        # 约束条件，包括等式约束和不等式约束
        cons = {'type': 'eq', 'fun': lambda w: 1 - np.dot(w, np.ones(m))}
        # cons = [{'type': 'ineq', 'fun': lambda w: w},  # 不等式约束默认符号为大于等于0
        #         {'type': 'ineq', 'fun': lambda w: upper - w},
        #         {'type': 'eq', 'fun': lambda w: 1 - np.dot(w, np.ones(m))}]

        # 优化求解
        res = minimize(obj_fun, w_0, method='trust-constr', constraints=cons, bounds=bound)

        # print(res.fun)
        print(res.success)
        # print(res.x)
        print(time.time() - start)
        return res.x

    @staticmethod
    def cvxpy_neu(config):

        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, l=100
        start = time.time()

        # 定义变量
        w_t = config["w_o"]
        r = config["r"]
        X = config["X"]
        w_b = config["w_b"]
        upper = config["upper"]
        l = config["l"]
        c = config["c"]
        m = len(w_t)  # 股票数

        w = cp.Variable(m)
        # print(f"w:{w},r:{r},w_t:{w_t},l:{l},w_b:{w_b},X:{X}")
        obj_fun = cp.Minimize(- w @ r + c * cp.norm(w - w_t, 1) + l * cp.norm((w - w_b) @ X) ** 2)

        # 约束条件，包括等式约束和不等式约束
        cons = [w >= 0,
                upper - w >= 0,
                w @ np.ones(m) == 1]

        # 把目标函数与约束传进Problem函数中
        prob = cp.Problem(obj_fun, cons)

        # 问题求解
        prob.solve()

        print("status:", prob.status)
        print(f"optimal value {prob.value}")
        print(f"optimal variable, w={w.value}")
        # print(res.x)
        print(time.time() - start)

        return w.value

    @staticmethod
    def cvxpy_with_exposure(config):

        # 原始参数：w_o, w_b, r, X, upper=0.02, c=0.006, l=100
        start = time.time()

        # 定义变量
        w_t = config["w_o"]
        r = config["r"]
        X = config["X"]
        w_b = config["w_b"]
        upper = config["upper"]
        c = config["c"]
        exposure = config["exposure"]
        m = len(w_t)  # 股票数

        w = cp.Variable(m)
        # print(f"w:{w},r:{r},w_t:{w_t},l:{l},w_b:{w_b},X:{X}")
        obj_fun = cp.Minimize(- w @ r + c * cp.norm(w - w_t, 1))

        # 约束条件，包括等式约束和不等式约束
        cons = [w >= 0,
                upper - w >= 0,
                w @ np.ones(m) == 1,
                # cp.norm((w - w_b) @ X) >= -exposure,
                cp.norm((w_b - w) @ X) <= exposure,
                cp.norm((w - w_b) @ X) <= exposure]

        # 把目标函数与约束传进Problem函数中
        prob = cp.Problem(obj_fun, cons)

        # 问题求解
        prob.solve()

        print("status:", prob.status)
        print(f"optimal value {prob.value}")
        print(f"optimal variable, w={w.value}")
        # print(res.x)
        print(time.time() - start)

        return w.value
