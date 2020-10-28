import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pymysql
from numpy import mean


def data_process(lags):
    # connect with database
    # conn = pymysql.connect(host='localhost', user='root', password='123456', database='pems', charset='utf8')
    # sql1 = "SELECT * FROM `bn4` WHERE datetime >= '%s' AND datetime <= '%s'" % ('2019-02-01 00:00:00', '2019-02-24 23:59:00')
    # sql2 = "SELECT * FROM `bn4` WHERE datetime >= '%s' AND datetime <= '%s'" % ('2019-02-25 00:00:00', '2019-02-28 23:59:00')
    #
    # # conn = pymysql.connect(host='localhost', user='root', password='123456', database='ramp_info', charset='utf8')
    # # sql1 = "SELECT * FROM `test` WHERE datetime >= '%s' AND datetime <= '%s'" % ('2018-06-01 00:00:00', '2018-07-09 23:59:00')
    # # sql2 = "SELECT * FROM `test` WHERE datetime >= '%s' AND datetime <= '%s'" % ('2018-07-10 00:00:00', '2018-07-20 23:59:00')
    # # import data
    # df1 = pd.read_sql(sql1, con=conn)
    # df2 = pd.read_sql(sql2, con=conn)
    # conn.close()

    df1 = pd.read_csv('pems_train.csv', encoding='utf-8').fillna(0)
    df2 = pd.read_csv('smooth_results.csv', encoding='utf-8').fillna(0)

    attr = 'flow'
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler