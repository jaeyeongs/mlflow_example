import numpy as np
import pandas as pd
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

class DataPreprocess:
    def __init__(self):
        self.df_path = os.getcwd() + r'\data\creditcard.csv'
        self.df = pd.read_csv(self.df_path)
        self.df = self.df.drop('Time', axis=1)
        # self.df = self.df.drop(self.df.columns[0:25], axis=1)
        # print(self.df)

    def data_split(self):
        # 정상 지점과 이상 징후 분할(정상 : 0, 이상 : 1)
        normal = self.df[self.df.Class == 0].sample(frac=0.5, random_state=2020).reset_index(drop=True)
        abnormal = self.df[self.df.Class == 1]

        # 정상 및 이상 데이터를 train / test/ validation 데이터 셋으로 분할
        # train / test / val = 0.6 / 0.2 / 0.2
        normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=2020)
        abnormal_train, abnormal_test = train_test_split(abnormal, test_size=0.2, random_state=2020)
        normal_train, normal_validate = train_test_split(normal_train, test_size=0.25, random_state=2020)
        abnormal_train, abnormal_validate = train_test_split(abnormal_train, test_size=0.25, random_state=2020)

        # 데이터 x, y 분할
        x_train = pd.concat((normal_train, abnormal_train))
        x_test = pd.concat((normal_test, abnormal_test))
        x_validate = pd.concat((normal_validate, abnormal_validate))

        y_train = np.array(x_train["Class"])
        y_test = np.array(x_test["Class"])
        y_validate = np.array(x_validate["Class"])

        x_train = x_train.drop("Class", axis=1)
        x_test = x_test.drop("Class", axis=1)
        x_validate = x_validate.drop("Class", axis=1)

        # data scale
        scaler = StandardScaler()
        scaler.fit(pd.concat((normal, abnormal)).drop("Class", axis=1))
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_validate = scaler.transform(x_validate)

        # print('Train set:\nx_train:{} \ny_train:{}'.format(x_train.shape, y_train.shape))
        # print('\nTest set:\nx_test:{} \ny_test:{}'.format(x_test.shape, y_test.shape))
        # print('\nVal set:\nx_train:{} \ny_validate:{}'.format(x_validate.shape, y_validate.shape))

        return x_train, x_validate, x_test, y_train, y_validate, y_test

