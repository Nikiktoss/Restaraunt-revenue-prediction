from catboost import CatBoostRegressor
import category_encoders as ce

import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae

import matplotlib.pyplot as plt


class RevenuePredictionModel:
    def __init__(self):
        self.model = CatBoostRegressor(n_estimators=250, loss_function="RMSE", learning_rate=0.524, depth=3,
                                       task_type='CPU', verbose=False)
        self.cat_encoder = ce.cat_boost.CatBoostEncoder()
        self.min_values = None
        self.max_values = None
        self.w = None

    @staticmethod
    def divide_data(data):
        now = datetime.datetime.now()

        cat_data = data[['City', 'City Group', 'Type']]

        num_data = data.drop(['Id', 'City', 'City Group', 'Type', 'Open Date'], axis=1)
        num_data['years_old'] = (now - pd.DatetimeIndex(data['Open Date'])).days // 365

        return num_data, cat_data

    def normalize_train_data(self, num_data):
        min_values = num_data.min()
        max_values = num_data.max()

        num_data = (num_data - min_values) / (max_values - min_values)
        self.min_values = min_values
        self.max_values = max_values

        return num_data

    def prepare_train_data(self, data):
        train_num_data, train_cat_data = self.divide_data(data.drop('revenue', axis=1))
        train_num_data = self.normalize_train_data(train_num_data)

        w = sum(data["revenue"])
        self.w = w

        y_train = np.sqrt(data["revenue"]) / np.sqrt(w)
        x_train = pd.concat([train_cat_data, train_num_data], axis=1)

        self.cat_encoder.fit(x_train, y_train)
        x_train = self.cat_encoder.transform(x_train)

        return x_train, y_train

    def fit_model(self, data):
        x_train, y_train = self.prepare_train_data(data)
        self.model.fit(x_train, y_train)

    def normalize_data(self, data):
        data = (data - self.min_values) / (self.max_values - self.min_values)
        return data

    def prepare_data(self, data):
        num_data, cat_data = self.divide_data(data)
        num_data = self.normalize_data(num_data)

        x = pd.concat([cat_data, num_data], axis=1)
        x = self.cat_encoder.transform(x)

        return x

    def predict_revenue(self, data):
        x = self.prepare_data(data)
        revenue = np.power(self.model.predict(x) * np.sqrt(self.w), 2)

        return revenue


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_valid_data = train_data.loc[123:, :]

train_data = train_data.iloc[:123, :]

train_cols = train_data.columns
test_valid_data = pd.DataFrame(np.array(test_valid_data), columns=train_cols)


revenue_model = RevenuePredictionModel()
revenue_model.fit_model(train_data)

y_test = test_valid_data['revenue']
y_predict = revenue_model.predict_revenue(test_valid_data.drop('revenue', axis=1))

print(f"R2 is {r2_score(y_test, y_predict)}")
cb_rmse = np.sqrt(mse(y_test, y_predict))
print(f"MAE is {mae(y_test, y_predict)}")
print("RMSE in y units:", np.mean(cb_rmse))

plt.plot(y_test, 'ro')
plt.plot(y_predict, 'go')
plt.show()
