class MY_PO_PARAMS:
    # 算法参数
    WEIGHT_UPPER = 0.05
    TRANSACTION_COST = 0.0006
    L = 0.5
    LAM = 0.5
    EXPOSURE = 0.17

    # CALC_METHOD = "calc_weight"
    # CALC_METHOD = "calc_weight_with_exposure"

    ''' 已有的优化工具 '''
    # CALC_METHOD = "cvxpy_without_risk_control"
    # CALC_METHOD = "cvxpy_neu"
    CALC_METHOD = "cvxpy_with_exposure"
    # CALC_METHOD = "SciOpt_neu"

    def to_str(self):
        return "\nCALCULATE WEIGHT METHOD: {}\nWEIGHT_UPPER: {}\nTRANSACTION_COST: {}" \
               "\nL: {}\nLAM: {}\nEXPOSURE: {}".format(self.CALC_METHOD, self.WEIGHT_UPPER,
                                                       self.TRANSACTION_COST, self.L, self.LAM, self.EXPOSURE)
