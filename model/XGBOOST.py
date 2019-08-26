from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

def fit_and_predict(inflow_data, outflow_data, inflow_ground_truth, outflow_ground_truth, deep, error_low, max_value):

    inflow_data_train, inflow_data_test, inflow_ground_truth_train, inflow_ground_truth_test = \
        train_test_split(inflow_data, inflow_ground_truth, test_size=0.1, random_state=1234565)

    outflow_data_train, outflow_data_test, outflow_ground_truth_train, outflow_ground_truth_test = \
        train_test_split(outflow_data, outflow_ground_truth, test_size=0.1, random_state=1234565)

    # bike 600
    # taxi 2500
    model = xgb.XGBRegressor(max_depth=15, learning_rate=0.001, n_estimators=deep, silent=False, objective='reg:gamma')
    #model_outflow = xgb.XGBRegressor(max_depth=15, learning_rate=0.001, n_estimators=1000, silent=False, objective='reg:gamma')


    model.fit(inflow_data_train, inflow_ground_truth_train)
    inflow_data_pred = model.predict(inflow_data_test)
    inflow_rmse = inflow_data_pred - inflow_ground_truth_test
    inflow_mape = inflow_data_pred - inflow_ground_truth_test
                  # / np.clip(inflow_ground_truth_test, error_low, None)

    model.fit(outflow_data_train, outflow_ground_truth_train)
    outflow_data_pred = model.predict(outflow_data_test)
    outflow_rmse = outflow_data_pred- outflow_ground_truth_test
    outflow_mape = outflow_data_pred - outflow_ground_truth_test
                  # / np.clip(outflow_ground_truth_test, error_low, None)

    flow_rmse = np.concatenate([inflow_rmse, outflow_rmse], axis=-1)
    flow_mape = np.concatenate([inflow_mape, outflow_mape], axis=-1)

    print('max', max_value)

    return np.mean(np.square(flow_rmse), axis=-1) ** 0.5, np.mean(np.abs(flow_mape), axis=-1)


"""
X = np.random.randint(1, 100, size=(100, 10))
y = np.random.randint(1, 50, size=(100, ))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

# booster [default=gbtree]
# 有两中模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree。


model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')

model.fit(X_train, y_train)
ans = model.predict(X_test)

print(np.mean(np.square(ans - y_test)) ** 0.5)
"""