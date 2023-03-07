import random

import keras
from keras.layers import Dense
import category_encoders as ce
import tensorflow

import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae


seed = 0
random.seed(seed)
np.random.seed(seed)
tensorflow.random.set_seed(seed)

# read data
train_data = pd.read_csv('train.csv')
test_valid_data = train_data.loc[123:, :]

train_data = train_data.iloc[:123, :]

train_cols = train_data.columns
test_valid_data = pd.DataFrame(np.array(test_valid_data), columns=train_cols)


def divide_data(data):
    cat_data = data[["City Group", "City"]]
    num_data = data.drop(["Id", "City", "City Group", "Type", "revenue"], axis=1)

    return num_data, cat_data


def edit_num_data(data):
    now = datetime.datetime.now()
    scale = preprocessing.Normalizer()

    data['years_old'] = (now - pd.DatetimeIndex(data['Open Date'])).days // 365
    data.drop("Open Date", axis=1, inplace=True)

    num_cols = data.columns
    num_data = pd.DataFrame(scale.fit_transform(data), columns=num_cols)

    return num_data


def create_model():
    model = keras.Sequential()
    model.add(Dense(80, input_dim=40, activation='relu'))

    model.add(Dense(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(120, activation='relu'))

    model.add(Dense(160))
    model.add(Dense(160))

    model.add(Dense(40))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


number_train_data, category_train_data = divide_data(train_data)
number_train_data = edit_num_data(number_train_data)

x_train = pd.concat([category_train_data, number_train_data], axis=1)
w = np.sqrt(sum(train_data["revenue"]))
y_train = train_data["revenue"] / w

cbe_encoder = ce.cat_boost.CatBoostEncoder()
cbe_encoder.fit(x_train, y_train)
x_train = cbe_encoder.transform(x_train)


revenue_model = create_model()
history = revenue_model.fit(x_train, y_train, epochs=70, verbose=False)

number_test_data, category_test_data = divide_data(test_valid_data)

number_test_data = edit_num_data(number_test_data)

x_test = pd.concat([category_test_data, number_test_data], axis=1)
x_test = cbe_encoder.transform(x_test)

y_test = test_valid_data["revenue"]

y_predict = revenue_model.predict(x_test) * w

print(f'R2 is {r2_score(y_test, y_predict)}')
cb_rmse = np.sqrt(mse(y_test, y_predict))
print(f"MAE is {mae(y_test, y_predict)}")
print("RMSE in y units:", np.mean(cb_rmse))
