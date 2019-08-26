# -*- coding:utf-8 -*-
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from statsmodels.tsa.stattools import adfuller as ADF
import math
import matplotlib.pyplot as plt
import numpy as np

def evaluate(arima_inflow_data, arima_inflow_ground_truth, arima_outflow_data, arima_outflow_ground_truth):

    rmse_list = []
    mape_list = []
    length = len(arima_outflow_ground_truth)

    for i in range(length):
        # todo get the data of grid i and truth i
        inflow_observations = pd.Series(arima_inflow_data[i])
        inflow_ground_truth = arima_inflow_ground_truth[i]

        outflow_observations = pd.Series(arima_outflow_data[i])
        outflow_ground_truth = arima_outflow_ground_truth[i]

        # todo get the p, d, q and fit model
        print('original in ADF result is', ADF(inflow_observations, 1))
        inflow_D_data = inflow_observations.diff(1).dropna()
        print('Diffenciate in ADF result is', ADF(inflow_D_data, 1))

        print('original out ADF result is', ADF(outflow_observations, 1))
        outflow_D_data = outflow_observations.diff(1).dropna()
        print('Diffenciate out ADF result is', ADF(outflow_D_data, 1))

        # todo forecast and calculate the error
        p = 0
        q = 0
        # 0 0 0.01600
        # 1 0 0.023
        # 1 1 0.026
        # 0 1 0.01964

        inflow_model = ARIMA(inflow_observations.values, (p, 1, q)).fit()
        inflow_result, _b, _c = inflow_model.forecast(1)
        inflow_loss_item = inflow_result - inflow_ground_truth

        outflow_model = ARIMA(outflow_observations.values, (p, 1, q)).fit()
        outflow_result, _b, _c = outflow_model.forecast(1)
        outflow_loss_item = outflow_result - outflow_ground_truth

        rmse_list.append(inflow_loss_item)
        rmse_list.append(outflow_loss_item)
        mape_list.append(inflow_loss_item)
        mape_list.append(outflow_loss_item)

    rmse_list = np.array(rmse_list)
    return np.mean(np.square(rmse_list)) ** 0.5, np.mean(np.abs(mape_list))


