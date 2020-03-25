import numpy as np


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
        if j == 1 or v_sorted[j - 1] - (sum_mur - z) / j > 0:
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


class ReferPO:
    def __init__(self):
        pass

    @staticmethod
    def olu(inputs):
        """
        paper: Online Lazy Updates for Portfolio Selection with Transaction Costs
        :param inputs: 算法的输入
                x_t:    相对价格向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_t = inputs['x_t']
        w_o = inputs['w_o']

        # iteration constant
        eps = 1e-4
        max_iter = 10000

        # algorithm constant
        beta = 0.01
        eta = 1
        gamma = 0.002
        alpha = eta * gamma

        # mu = 1e-4
        # maxmu = 1e10
        # rho = 1.02
        # p = np.zeros(len(w_o))

        # initialize
        n = len(w_o)
        w = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        k = 0
        while (np.linalg.norm(w - w_o - z) > eps or np.linalg.norm(np.sum(w) - 1) > eps) and k < max_iter:
            # 原文做法
            # iterate p(k+1), z(k+1), u(k+1)
            part1 = -1 * eta / ((beta + 1) * np.sum(w_o * x_t)) * x_t
            part2 = w_o
            part3 = beta / (beta + 1) * z
            part4 = -1 * beta / (beta + 1) * u
            w = simplex_projection(part1 + part2 + part3 + part4)

            threshold = alpha / beta
            proj_vec = w - w_o + u
            for i in range(n):
                z[i] = min(0.0, proj_vec[i] + threshold) + max(0.0, proj_vec[i] - threshold)

            u = u + (w - w_o - z)

            # # 自己设计解法
            # w = (eta + mu * (w_o + z) - p - w_o) / (1 + mu)
            #
            # threshold = alpha / mu
            # proj_vec = w - w_o + p / mu
            # for i in range(n):
            #     z[i] = min(0.0, proj_vec[i] + threshold) + max(0.0, proj_vec[i] - threshold)
            #
            # p = p + mu * (w - z - w_o)
            # mu = min(mu * rho, maxmu)

            k = k + 1
            if k % 200 == 0 or (np.linalg.norm(w - w_o - z) <= eps and np.linalg.norm(np.sum(w) - 1) <= eps):
                ret = np.sum(w * x_t)
                target_1 = -eta * np.log(ret)
                target_2 = alpha * np.linalg.norm((w - w_o), ord=1)
                target_3 = 1 / 2 * np.linalg.norm(w - w_o)
                total = target_1 + target_2 + target_3

                print("iter {} return {} target_1 {} target_2 {} target_3 {}"
                      " total_value {} norm_1 {} norm_2 {}".format(k, ret, target_1, target_2, target_3, total,
                                                                   np.linalg.norm(w - w_o - z),
                                                                   np.linalg.norm(np.sum(w) - 1)))

        return w

    @staticmethod
    def olmar(inputs):
        """
        paper: On-Line Portfolio Selection with Moving Average Reversion
        :param inputs: 算法输入
                x_pred: 相对价格预测向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_pred = inputs['x_pred']
        w_o = inputs['w_o']

        # algorithm constant
        eps = 10
        # omega = 5

        x_pred_avg = np.mean(x_pred)
        lam = max(0, (eps - np.sum(w_o * x_pred)) / (np.linalg.norm(x_pred - x_pred_avg) ** 2))
        w = w_o + lam * (x_pred - x_pred_avg)
        w = simplex_projection(w)

        return w

    @staticmethod
    def rmr(inputs):
        """
        paper: Robust Median Reversion Strategy for Online Portfolio Selection
        :param inputs: 算法输入
                x_pred: 相对价格预测向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_pred = inputs['x_pred']
        w_o = inputs['w_o']

        # algorithm constant
        eps = 5
        # eps = inputs['eps']
        # omega = 5

        x_pred_avg = np.mean(x_pred)
        lam = min(0, (np.sum(w_o * x_pred) - eps) / (np.linalg.norm(x_pred - x_pred_avg) ** 2))
        w = w_o - lam * (x_pred - x_pred_avg)
        w = simplex_projection(w)

        return w

    @staticmethod
    def sspo(inputs):
        """
        paper: Short-term Sparse Portfolio Optimization Based on Alternating Direction Method of Multipliers
        :param inputs: 算法输入
                x_pred: 相对价格预测向量，size = 1 * stocks
                w_o:    原来的组合权重，size = 1 * stocks
        :return: w(t+1)
        """
        # get inputs
        x_pred = inputs['x_pred']
        w_o = inputs['w_o']

        # iteration constant
        eps = 1e-4
        max_iter = 10000

        # algorithm constant
        # omega = 5
        lam = 0.5
        gamma = 0.01
        eta = 0.005
        kxi = 500
        phi = -1.1 * np.log(x_pred) - 1

        # initialize
        n = len(w_o)
        w = np.copy(w_o)
        g = np.copy(w_o)
        rho = np.zeros(n)

        o = 1
        while np.abs(np.sum(w) - 1) > eps and o < max_iter:
            part_l = lam / gamma * np.identity(n) + eta
            part_r = (lam / gamma) * g + (eta - rho) - phi
            w = np.dot(np.linalg.inv(part_l), part_r)
            part_l = np.sign(w)
            part_r = np.abs(w) - gamma * np.ones(n)
            for i in range(n):
                part_r[i] = part_r[i] if part_r[i] > 0 else 0
            g = part_l * part_r
            rho = rho + eta * (np.sum(w) - 1)

            o = o + 1
            if o % 200 == 0 or np.abs(np.sum(w) - 1) <= eps:
                R_t = np.dot(phi, w)
                sparsity = np.size(np.where(w != 0))
                total = -R_t + lam * np.linalg.norm(w, ord=1)
                print("iter {} return {} sparity {} total_value {}".format(o, R_t, sparsity, total))

        w = simplex_projection(w * kxi)
        return w

    @staticmethod
    def eg(inputs):
        # get inputs
        eta = 0.05
        x_t = inputs['x_t']
        w_o = inputs['w_o']
        if np.dot(w_o, x_t):
            theta_t = x_t / np.dot(w_o, x_t)
            w = w_o * np.exp(eta * theta_t)
            return w
        else:
            return np.ones_like(w_o) / w_o.size

    @staticmethod
    def pamr(inputs):
        # get inputs
        eps = 0.5
        x_t = inputs['x_t']
        w_o = inputs['w_o']
        tau = max(0.0, np.dot(w_o * x_t) - eps / np.linalg.norm(x_t - np.mean(x_t)))
        w = w_o - tau * (x_t - np.mean(x_t))
        return w
