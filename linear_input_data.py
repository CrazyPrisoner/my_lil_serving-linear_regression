import pandas as pd
import numpy
from sklearn.model_selection import train_test_split


def input_data():
    df = pd.read_csv('/home/deka/Desktop/datasets/Dataset1_dp.csv')
    df.drop('dateandtime',axis=1,inplace=True)
    df.drop('Status',axis=1,inplace=True)
    df.dropna(inplace=True)
    z = df.values[:, 0]
    v = df.values[:, 1]
    x_tr, x_te, y_tr, y_te = train_test_split(z,v,test_size=0.25)
    train_X = numpy.asarray(x_tr)
    train_Y = numpy.asarray(y_tr)
    test_X = numpy.asarray(x_te)
    test_Y = numpy.asarray(y_te)
    return z, v, train_X, train_Y, test_X, test_Y
