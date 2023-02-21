from catboost import CatBoostRegressor, Pool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse, r2_score


# read data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_valid_data = train_data.loc[123:, :]

train_data = train_data.iloc[:122, :]

train_cols = train_data.columns
test_valid_data = pd.DataFrame(np.array(test_valid_data), columns=train_cols)


# dividing DataFrame into two frames: frame with numeric data and frame with category data
def divide_data(data):
    cat_data = data[['Type']]

    now = datetime.datetime.now()
    num_data = data.drop(['City', 'City Group', 'Type', 'Open Date', 'revenue'], axis=1)
    num_data['years_old'] = now.year - pd.DatetimeIndex(data['Open Date']).year

    return num_data, cat_data


number_train_data, category_train_data = divide_data(train_data)

y_train = train_data["revenue"]
x_train = pd.concat([category_train_data, number_train_data], axis=1)

cb = CatBoostRegressor(n_estimators=300, loss_function="RMSE", learning_rate=0.4, depth=3, task_type='CPU',
                       random_state=17, verbose=False)

pool_train = Pool(x_train, y_train, cat_features=['Type'])
cb.fit(pool_train)

number_test_data, category_test_data = divide_data(test_valid_data)

y_test = np.array(test_valid_data["revenue"])
x_test = pd.concat([category_test_data, number_test_data], axis=1)

pool_test = Pool(x_test, cat_features=['Type'])
y_predict = cb.predict(pool_test)
print(r2_score(y_test, y_predict))

cb_rmse = np.sqrt(mse(y_test, y_predict))
print("RMSE in y units:", np.mean(cb_rmse))
