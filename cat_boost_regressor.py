from catboost import CatBoostRegressor
import category_encoders as ce

import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae

import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_valid_data = train_data.loc[123:, :]

train_data = train_data.iloc[:123, :]

train_cols = train_data.columns
test_valid_data = pd.DataFrame(np.array(test_valid_data), columns=train_cols)


def divide_data(data):
    now = datetime.datetime.now()

    cat_data = data[['City', 'City Group', 'Type']]

    num_data = data.drop(['Id', 'City', 'City Group', 'Type', 'Open Date', 'revenue'], axis=1)
    num_data['years_old'] = (now - pd.DatetimeIndex(data['Open Date'])).days // 365

    return num_data, cat_data


def normalize_train_data(data):
    min_values = data.min()
    max_values = data.max()

    data = (data - min_values) / (max_values - min_values)
    return data, min_values, max_values


def normalize_test_data(data, min_values, max_values):
    data = (data - min_values) / (max_values - min_values)
    return data


train_num_data, train_cat_data = divide_data(train_data)
train_num_data, min_norm, max_norm = normalize_train_data(train_num_data)

w = sum(train_data["revenue"])
y_train = np.sqrt(train_data["revenue"]) / np.sqrt(w)
x_train = pd.concat([train_cat_data, train_num_data], axis=1)

cbe_encoder = ce.cat_boost.CatBoostEncoder()
cbe_encoder.fit(x_train, y_train)
x_train = cbe_encoder.transform(x_train)


cb = CatBoostRegressor(n_estimators=250, loss_function="RMSE", learning_rate=0.524, depth=3, task_type='CPU',
                       verbose=False)
cb.fit(x_train, y_train)


test_num_data, test_cat_data = divide_data(test_valid_data)
test_num_data = normalize_test_data(test_num_data, min_norm, max_norm)

x_test = pd.concat([test_cat_data, test_num_data], axis=1)
x_test = cbe_encoder.transform(x_test)
y_test = test_valid_data['revenue']

y_predict = np.power(cb.predict(x_test) * np.sqrt(w), 2)

print(f"R2 is {r2_score(y_test, y_predict)}")
cb_rmse = np.sqrt(mse(y_test, y_predict))
print(f"MAE is {mae(y_test, y_predict)}")
print("RMSE in y units:", np.mean(cb_rmse))

plt.plot(y_test, 'ro')
plt.plot(y_predict, 'go')
plt.show()
