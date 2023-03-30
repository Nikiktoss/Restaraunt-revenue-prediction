from catboost import CatBoostRegressor
import category_encoders as ce
from fpdf import FPDF, HTMLMixin

import pandas as pd
import numpy as np
import datetime
import random

import pickle


seed = 0
random.seed(seed)
np.random.seed(seed)


class NormalizeDataClass:
    def __init__(self):
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

    def prepare_train_data(self, data=pd.read_csv('train.csv')):
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
        return x_train, y_train

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
        return x


class PDF(FPDF, HTMLMixin):
    cities = {'İstanbul': 'Istanbul', 'Diyarbakır': 'Diyarbakir', 'İzmir': 'Izmir', 'Elazığ': 'Elazıg',
              'Eskişehir': 'Eskisehir', 'Şanlıurfa': 'Sanliurfa', 'Uşak': 'Usak', 'Muğla': 'Mugla',
              'Kırklareli': 'Kirklareli', 'Karabük': 'Karabuk', 'Tekirdağ': 'Tekirdag', 'Balıkesir': 'Balikesir',
              'Aydın': 'Aydin', 'Kütahya': 'Kutahya'}

    def __init__(self, orientation='P', unit='mm', format='A4', rgb_color=(17, 14, 173)):
        super().__init__(orientation, unit, format)
        self.rgb_color = rgb_color

    def create_table(self, columns, values):
        num_of_tables = (len(columns) // 10) + 1

        for i in range(num_of_tables):
            if i == num_of_tables - 1:
                header = self.write_table_header(columns[i * 10:])
                row = self.write_table_row(values[i * 10:])
            else:
                header = self.write_table_header(columns[i * 10: (i + 1) * 10])
                row = self.write_table_row(values[i * 10: (i + 1) * 10])

            if self.will_page_break(20) is True:
                self.add_page()

            self.write_html(f"""<table width="100%" border="1">
                  <thead>
                    <tr>
                      {header}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      {row}
                    </tr>
                  </tbody>
                </table>""")

    @staticmethod
    def write_table_header(columns):
        header = ""

        for column in columns:
            header += f"<th>{column}</th>\n"

        return header

    @classmethod
    def write_table_row(cls, values):
        row = ""
        num_of_symbols = 2

        if len(values) < 10:
            num_of_symbols = 5

        for value in values:
            if value in cls.cities:
                value = cls.cities[value]
                row += f"<td align='center'>{value}</td>\n"
            elif type(value) != str:
                row += f"<td align='center'>{float(value):.{num_of_symbols}f}</td>\n"
            else:
                row += f"<td align='center'>{value}</td>\n"

        return row

    def block_table(self, table_name, columns, values, table_size=16):
        self.set_text_color(self.rgb_color)
        self.set_font("Arial", size=16)

        self.cell(200, 5, txt=f"{table_name}", ln=1)

        self.set_text_color(0, 0, 0)
        self.set_font("Arial", size=table_size)
        self.create_table(columns, values)

    def block_result(self, result):
        self.set_font("Arial", size=18)
        self.set_text_color(self.rgb_color)

        self.cell(200, 15, txt="Final result:", ln=1)

        self.set_text_color(0, 0, 0)
        self.cell(200, 10, txt=f"{result}", ln=1)

    def block_title(self, title):
        self.set_font("Arial", size=16)
        self.set_text_color(self.rgb_color)

        self.cell(200, 20, txt=f"{title}", align="C", ln=1)


def generate_pdf_file(pdf_file, data):
    num_data, cat_data = normalizer.divide_data(data)

    normalize_test_data = normalizer.prepare_data(data)
    normalize_cat_data = normalize_test_data[['City', 'City Group', 'Type']]
    normalize_num_data = normalize_test_data.drop(['City', 'City Group', 'Type'], axis=1)

    pdf_file.add_page()
    pdf_file.set_font("Arial", size=18)
    pdf_file.cell(200, 10, txt="Restaurant revenue prediction result", align="C", ln=1)

    pdf_file.block_title("Input data")
    pdf_file.block_table("Category data:", cat_data.columns, cat_data.values[0])
    pdf_file.block_table("Numeric data:", num_data.columns, num_data.values[0], 10)

    pdf_file.block_title("Normalize data")
    pdf_file.block_table("Category data:", normalize_cat_data.columns, normalize_cat_data.values[0])
    pdf_file.block_table("Numeric data:", normalize_num_data.columns, normalize_num_data.values[0], 10)

    new_x = normalizer.prepare_data(data)
    revenue = np.power(cb.predict(new_x) * np.sqrt(normalizer.w), 2)
    pdf_file.block_result(revenue[0])

    return pdf_file.output()


train_data = pd.read_csv('train.csv')
train_data = train_data.iloc[:123, :]

normalizer = NormalizeDataClass()
x_tr, y_tr = normalizer.fit_model(train_data)
pdf = PDF()


if __name__ == '__main__':
    cb = CatBoostRegressor(n_estimators=195, loss_function="RMSE", learning_rate=0.5, depth=3, task_type='CPU',
                           verbose=False)

    cb.fit(x_tr, y_tr)

    with open('model.pkl', 'wb') as f:
        pickle.dump(cb, f)
