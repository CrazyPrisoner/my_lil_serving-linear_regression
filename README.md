<h1> Linear Regression, Tensorflow Serving </h1>


<h2> Script to visualize data and feed model. </h2>

<p> Import packages </p>

    import numpy # for arrays
    import pandas # for dataframe
    import matplotlib.pyplot as plt # for graphics
    from sklearn.model_selection import train_test_split # for dived data on train and test data

<p> Import data </P>

    def import_data():
        dataframe = pandas.read_csv('/home/deka/Desktop/datasets/lil_dataset.csv')
        return dataframe
        
<p> Information about data and visualize it/p>
 
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
        
<p> Devide data, om train data and test data for feed model, need 2 parameters: N and EGT </p>

    def input_data():
        df = import_data()
        df.dropna(inplace=True)
        z = df.values[:, 0]
        v = df.values[:, 1]
        x_tr, x_te, y_tr, y_te = train_test_split(z,v,test_size=0.25)
        train_X = numpy.asarray(x_tr)
        train_Y = numpy.asarray(y_tr)
        test_X = numpy.asarray(x_te)
        test_Y = numpy.asarray(y_te)
        return z, v, train_X, train_Y, test_X, test_Y

