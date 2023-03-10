from catboost import CatBoostRegressor
import category_encoders as ce

import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae


# read data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_valid_data = train_data.loc[123:, :]

train_data = train_data.iloc[:123, :]

train_cols = train_data.columns
test_valid_data = pd.DataFrame(np.array(test_valid_data), columns=train_cols)


# dividing DataFrame into two frames: frame with numeric data and frame with category data
def divide_data(data):
    now = datetime.datetime.now()
    scale = preprocessing.Normalizer()

    cat_data = data[['City', 'City Group', 'Type']]

    num_data = data.drop(['City', 'City Group', 'Type', 'Open Date', 'revenue'], axis=1)
    num_data['years_old'] = (now - pd.DatetimeIndex(data['Open Date'])).days // 365

    num_cols = num_data.columns
    num_data = pd.DataFrame(scale.fit_transform(num_data), columns=num_cols)

    return pd.concat([cat_data, num_data], axis=1)


x_train = divide_data(train_data)
w = sum(train_data["revenue"])
y_train = np.sqrt(train_data["revenue"]) / np.sqrt(w)

cbe_encoder = ce.cat_boost.CatBoostEncoder()
cbe_encoder.fit(x_train, y_train)
x_train = cbe_encoder.transform(x_train)


cb = CatBoostRegressor(n_estimators=250, loss_function="RMSE", learning_rate=0.50905, depth=3, task_type='CPU',
                       random_state=38, verbose=False)
cb.fit(x_train, y_train)


x_test = divide_data(test_valid_data)
y_test = test_valid_data['revenue']
x_test = cbe_encoder.transform(x_test)

y_predict = np.power(cb.predict(x_test) * np.sqrt(w), 2)

print(r2_score(y_test, y_predict))
cb_rmse = np.sqrt(mse(y_test, y_predict))
print(f"MAE is {mae(y_test, y_predict)}")
print("RMSE in y units:", np.mean(cb_rmse))
