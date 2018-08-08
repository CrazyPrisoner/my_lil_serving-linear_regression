<h1> Linear Regression, Tensorflow Serving </h1>


<h2> Script to visualize data and feed model. </h2>

<p> Import packages </p>

    import numpy # for arrays
    import pandas # for dataframe
    import matplotlib.pyplot as plt # for graphics
    from sklearn.model_selection import train_test_split # for dived data on train and test data

<p> Import data </P>

    def import_data():
        dataframe = pandas.read_csv('/home/deka/Desktop/datasets/lil_dataset.csv') # import dataset the convert it to dataframe
        return dataframe
        
<p> Information about data and visualize it </p>
 
     def info_data():
        data = import_data() # import data
        print(data.info(),"\n") # show information about dataframe
        print(data.head(10),"\n") # show first 10 values
        print(data.corr(),"\n") # correlation

        plt.plot(data['N'])
        plt.ylabel('Parameter N')
        plt.show()
        plt.plot(data['EGT'],color='red')
        plt.ylabel('Parameter EGT')
        plt.show()
        plt.plot(data['WF'],color='green')
        plt.ylabel('Parameter WF')
        plt.show()
        
<p> Devide data, on train data and test data for feed model, need 2 parameters: N and EGT </p>

    def input_data():
        df = import_data() # import dataframe
        df.dropna(inplace=True) # delete NaN values
        z = df.values[:, 1] # Parameter N
        v = df.values[:, 2] # Parameter EGT
        x_tr, x_te, y_tr, y_te = train_test_split(z,v,test_size=0.25) #divide data, 75% train data, 25% test data.
        train_X = numpy.asarray(x_tr) # convert ndarray in array
        train_Y = numpy.asarray(y_tr) # convert ndarray in array
        test_X = numpy.asarray(x_te) # convert ndarray in array
        test_Y = numpy.asarray(y_te) # convert ndarray in array
        return z, v, train_X, train_Y, test_X, test_Y

