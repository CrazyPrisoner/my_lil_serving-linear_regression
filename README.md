 ############## linear_input_data.py ############## 
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split


def input_data():
    df = pd.read_csv('/home/deka/Desktop/datasets/lil_dataset.csv')
    df.drop('dateandtime',axis=1,inplace=True)
    df.drop('Status',axis=1,inplace=True)
    df.dropna(inplace=True)
    z = df.values[:, 0] # first parametr, it's a N
    v = df.values[:, 1] # second parametr, it's a EGT
    x_tr, x_te, y_tr, y_te = train_test_split(z,v,test_size=0.25) # I devide on train and test data
    train_X = numpy.asarray(x_tr) # simple numpy array
    train_Y = numpy.asarray(y_tr) # simple numpy array
    test_X = numpy.asarray(x_te) # simple numpy array
    test_Y = numpy.asarray(y_te) # simple numpy array
    return z, v, train_X, train_Y, test_X, test_Y
    
  ############## 
