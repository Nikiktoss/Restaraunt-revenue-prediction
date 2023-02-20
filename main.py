import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
from sklearn.decomposition import PCA


# read data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
valid_data = pd.read_csv('sampleSubmission.csv')


# dividing DataFrame into two frames: frame with numeric data and frame with category data
def divide_data(data):
    now = datetime.datetime.now()
    cat_data = data[['City', 'City Group', 'Type', 'Open Date']]
    num_data = data.drop(['City', 'City Group', 'Type', 'Open Date'], axis=1)
    num_data['years_old'] = now.year - pd.DatetimeIndex(data['Open Date']).year

    return num_data, cat_data


def edit_category_data(data):
    le = preprocessing.LabelEncoder()
    data['City Group'] = le.fit_transform(data['City Group'])
    data = pd.concat([data, pd.get_dummies(data.Type)], axis=1)
    data.drop(['Open Date', 'City', 'Type'], axis=1, inplace=True)

    return data


def edit_number_data(data, drop_cols):
    data.drop(drop_cols, axis=1, inplace=True)
    x = np.log(data[['years_old']])
    y = np.sqrt(data.drop(['years_old'], axis=1))
    y_columns = y.columns
    sc = preprocessing.StandardScaler()
    y = sc.fit_transform(y)
    y = pd.DataFrame(y, columns=y_columns)
    return pd.concat([y, x], axis=1)


def draw_plots(data, revenue):
    features = data.drop(['years_old', 'Id'], axis=1)
    features_cols = features.columns
    features = pd.DataFrame(np.sqrt(features), columns=features_cols)
    revenue = np.sqrt(revenue)
    counter = 0
    for i in range(4):
        figure, axis = plt.subplots(2, 5, figsize=(14, 6))
        figure.tight_layout(h_pad=3)
        for j in range(2):
            for k in range(5):
                axis[j, k].plot(features[features.columns[counter]], revenue, "ro")
                axis[j, k].set_title(features.columns[counter])
                counter += 1
                if counter == len(features.columns):
                    break
            if counter == len(features.columns):
                break
        plt.show()


def get_result_data(num_data, cat_data):
    x_values_training = pd.concat([cat_data, num_data], axis=1)
    return x_values_training


def decrease_x_using_pca(x, func):
    x = func.transform(x)

    return x


pca = PCA(0.98)

number_data, category_data = divide_data(train_data)
number_data.drop('revenue', axis=1, inplace=True)
revenue_data = np.log(train_data['revenue'])
category_data = edit_category_data(category_data)

# draw_plots(number_data, revenue_data)
# columns to drop, it goes from plots

drop = ['P7', 'P9', 'P10', 'P13', 'P18', 'P34', 'P36']

number_data = edit_number_data(number_data, drop)
x_train = get_result_data(number_data, category_data)
pca.fit(x_train)
x_train = decrease_x_using_pca(x_train, pca)

components = len(x_train[0])
gbr = GradientBoostingRegressor(max_depth=3, n_estimators=components, criterion='friedman_mse', random_state=42)
gbr.fit(x_train, revenue_data)


number_test_data, category_test_data = divide_data(test_data)
category_test_data = edit_category_data(category_test_data)
category_test_data.drop('MB', axis=1, inplace=True)

number_test_data = edit_number_data(number_test_data, drop)
x_test = get_result_data(number_test_data, category_test_data)
x_test = decrease_x_using_pca(x_test, pca)


y_predict = gbr.predict(x_test)
y_test = valid_data['Prediction']
y_predict = np.round(np.exp(y_predict), 2)
mse_friedman = mean_squared_error(y_predict, y_test)



# print(f"MSE is {mse_friedman}")
# print(f"RMSE is {np.sqrt(mse_friedman)}")
# print(f"r2 is {r2_score(y_test, y_predict)}")

# plt.plot(y_test[0:100], 'go')
# plt.plot(y_predict[0:100], 'ro')
# plt.show()
