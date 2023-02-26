import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score
import category_encoders as ce


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_valid_data = train_data.loc[123:, :]

train_data = train_data.iloc[:123, :]
train_types = train_data["Type"].unique()

train_cols = train_data.columns
test_valid_data = pd.DataFrame(np.array(test_valid_data), columns=train_cols)


def divide_data(data):
    cat_data = data[["City", "Open Date", "City Group", "Type"]]
    num_data = data.drop(["Id", "City", "City Group", "Type", "revenue", "P4", "P6", "P9", "P13", "P14", "P15", "P17",
                          "P19", "P25", "P34"], axis=1)

    return num_data, cat_data


def edit_num_data(data):
    now = datetime.datetime.now()
    scale = preprocessing.Normalizer()

    data['years_old'] = (now - pd.DatetimeIndex(data['Open Date'])).days // 365
    data.drop("Open Date", axis=1, inplace=True)

    num_cols = data.columns
    num_data = pd.DataFrame(scale.fit_transform(data), columns=num_cols)

    return num_data


number_train_data, category_train_data = divide_data(train_data)
number_train_data = edit_num_data(number_train_data)

x_train = pd.concat([category_train_data, number_train_data], axis=1)
y_train = np.sqrt(train_data[["revenue"]])

cbe_encoder = ce.cat_boost.CatBoostEncoder()
cbe_encoder.fit(x_train, y_train)
x_train = cbe_encoder.transform(x_train)

lr = LinearRegression()
lr.fit(x_train, y_train)

number_test_data, category_test_data = divide_data(test_valid_data)
number_test_data = edit_num_data(number_test_data)

x_test = pd.concat([category_test_data, number_test_data], axis=1)
x_test = cbe_encoder.transform(x_test)

y_test = test_valid_data["revenue"]

y_predict = np.power(lr.predict(x_test), 2)

print(r2_score(y_test, y_predict))
cb_rmse = np.sqrt(mse(y_test, y_predict))
print("RMSE in y units:", np.mean(cb_rmse))
