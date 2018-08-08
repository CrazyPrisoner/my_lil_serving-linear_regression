import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def import_data():
    dataframe = pandas.read_csv('/home/deka/Desktop/datasets/lil_dataset.csv')
    return dataframe

def info_data():
    data = import_data()
    print(data.info(),"\n")
    print(data.head(10),"\n")
    print(data.corr(),"\n")

    plt.plot(data['N'])
    plt.ylabel('Parameter N')
    plt.show()
    plt.plot(data['EGT'],color='red')
    plt.ylabel('Parameter EGT')
    plt.show()
    plt.plot(data['WF'],color='green')
    plt.ylabel('Parameter WF')
    plt.show()


def input_data():
    df = import_data()
    df.dropna(inplace=True)
    z = df.values[:, 1]
    v = df.values[:, 2]
    x_tr, x_te, y_tr, y_te = train_test_split(z,v,test_size=0.25)
    train_X = numpy.asarray(x_tr)
    train_Y = numpy.asarray(y_tr)
    test_X = numpy.asarray(x_te)
    test_Y = numpy.asarray(y_te)
    return z, v, train_X, train_Y, test_X, test_Y
