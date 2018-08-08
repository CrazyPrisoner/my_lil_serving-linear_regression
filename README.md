<h1> Linear Regression, Tensorflow Serving </h1>


<h2> Script to visualize data and feed model. </h2>

<p> Import packages. </p>

    import numpy # for arrays
    import pandas # for dataframe
    import matplotlib.pyplot as plt # for graphics
    from sklearn.model_selection import train_test_split # for dived data on train and test data

<p> Import data. </P>

    def import_data():
        dataframe = pandas.read_csv('/home/deka/Desktop/datasets/lil_dataset.csv') # path to dataset
        return dataframe
        
<p> Information about data and visualize it. </p>
 
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
        
<p> Devide data, on train data and test data for feed model, need 2 parameters: N and EGT. </p>

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


 <h2> Script to train model and save it.</h2>
 <p>Run in like this : python own_model_serving01.py /home/deka/Desktop/test_tensorflow_serving/test_serving_model3/. Need give path to save model.</p>
 
 
 
        from __future__ import print_function
        import os
        import sys # for console commands
        import numpy # for arrays
        rng = numpy.random # random values
        import tensorflow as tf # for create model
        import matplotlib.pyplot as plt # for graphics
        from linear_input_data import  * # script linear_input_data to feed and visualization

        tf.app.flags.DEFINE_integer('training_iteration', 1000,
                                    'number of training iterations.') # training epochs
        tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.') model version need to save it
        tf.app.flags.DEFINE_string('work_dir', '/home/deka/Desktop/test_tensorflow_serving/test_serving_model3', 'Working directory.') # work directory
        FLAGS = tf.app.flags.FLAGS


        def main(_):
            if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
                print('Usage: mnist_export.py [--training_iteration=x] '
                  '[--model_version=y] export_dir') # path to save trained model
                sys.exit(-1)
            if FLAGS.training_iteration <= 0:
                print('Please specify a positive value for training iteration.') # for iteration model
                sys.exit(-1) 
            if FLAGS.model_version <= 0:
                print('Please specify a positive value for version number.') # for save model
                sys.exit(-1)

            # info about our data we run it from linear_input_data, linear_input_data.info_data
            info_data()
            
 <p> Graphics, we call from linear_input_data.py . </p>

![parameter_egt](https://user-images.githubusercontent.com/37526996/43833319-2ddab504-9b2c-11e8-9786-9a2aad1e5193.png)
![parameter_wf](https://user-images.githubusercontent.com/37526996/43833327-30959caa-9b2c-11e8-9d72-4add9be08ee4.png)
![parameter_n](https://user-images.githubusercontent.com/37526996/43833329-32138e48-9b2c-11e8-8efc-f0ed81223e66.png)

            
            
  <p> Constuct model and preapare data. </p>
  
            # Hyperparameters
            learning_rate = 0.01 # learning speed
            display_step = 50 # display cost, weights, bias
            
            sess = tf.InteractiveSession()
            x,y,train_x,train_y,test_x,test_y = input_data() # take data from linear_input_data, linear_input_data.input_data()
            
            # Trainin example, as requested (Issue #1)
            train_X = numpy.asarray(train_x) # for train model
            train_Y = numpy.asarray(train_y) # for train model
            n_samples = train_X.shape[0]
            
            # Trainin example, as requested (Issue #1)
            test_X = numpy.asarray(test_x) # for test model
            test_Y = numpy.asarray(test_y) # for test model
            
            # tf Graph Input
            X = tf.placeholder(dtype=tf.float32,name="X") # placeholder to feed model, like a input layer
            Y = tf.placeholder(dtype=tf.float32,name="Y") # output layer, supervized learning
            
            # Set model weights
            W = tf.Variable(rng.randn(), name="weight") # weights
            b = tf.Variable(rng.randn(), name="bias") # bias
            sess.run(tf.global_variables_initializer()) # initialize all variables
            
            # Construct a linear model
            pred = tf.add(tf.multiply(X, W), b, name="prediction") # for prediction value
            # Mean squared error
            cost = tf.reduce_sum(tf.pow(pred-Y, 2),name="cost_func")/(2*n_samples) # cost function
            # Gradient descent
            #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # optimizer to learn
            
<p> Train model. </p>
            
            # Train model
            print('Training model...')
            for epoch in range(FLAGS.training_iteration):
                for (x, y) in zip(train_X, train_Y):
                    sess.run(optimizer, feed_dict={X: x, Y: y}) # optimize weights and bias
                    # Display logs per epoch step
                    if (epoch+1) % display_step == 0:
                        cosst = sess.run(cost, feed_dict={X: train_X, Y:train_Y}) # show information about cost, weights, bias every 50 steps
                print("Optimization Finished!")
                training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y}) # every optmization
                print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
                predict = sess.run(pred, feed_dict={X: train_X}) # for prediction
                print("Prediction :", predict)
            
            
<p> Test model. Predcit vizualization. </p> 
            
            # Test our model
            predict_test = sess.run(pred, feed_dict={X: test_X}) # test prediction with test data
            plt.plot(predict_test,color='green') # Predicted line
            plt.ylabel('Parameter EGT')
            plt.plot(test_Y,color='blue') # Test line
            plt.show()
            
![tested_model](https://user-images.githubusercontent.com/37526996/43833309-2343d2d8-9b2c-11e8-8714-137f35c6ed25.png)

<p> Saving model path, version, name. Saving model builder. </p>

            export_path_base = sys.argv[-1] # save model path
            export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(16))) # folder where will be saved model
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            
            
            
<p> Creating signature map. </p>


            # Build the signature_def_map.
            regression_inputs = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
            regression_outputs_prediction = tf.saved_model.utils.build_tensor_info(pred) # Save predcition function

            regression_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.REGRESS_INPUTS:regression_inputs
            },
            outputs={
                tf.saved_model.signature_constants.REGRESS_OUTPUTS:regression_outputs_prediction,
            },
            method_name=tf.saved_model.signature_constants.REGRESS_METHOD_NAME
            ))

            tensor_info_x = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take vost function
            tensor_info_y = tf.saved_model.utils.build_tensor_info(cost) # Save cost function

            prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_value':tensor_info_x},
            outputs={'output_value':tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_value':
                    prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    regression_signature,
            },
            legacy_init_op = legacy_init_op)

<p> Saving model. </p>

            builder.save()
            print("Done exporting!")

        if __name__ == '__main__':
            tf.app.run()

<h2> Run server. </h2>
<p> Run server like this: ```tensorflow_model_server --port=6660 --model_name=deka --model_base_path=/home/deka/Desktop/test_tensorflow_serving/test_serving_model3/```. </p>
    
<p>port="need_give_port", model_name="give_own_name_for_your_model", model_base_path="give_path_to_your_model".</p>


<p> If you run successfully, you can see this, in your command line. </p>

        2018-08-08 17:04:00.209111: I tensorflow_serving/model_servers/main.cc:153] Building single TensorFlow model file config:  model_name: deka model_base_path: /home/deka/Desktop/test_tensorflow_serving/test_serving_model3/
        2018-08-08 17:04:00.209486: I tensorflow_serving/model_servers/server_core.cc:459] Adding/updating models.
        2018-08-08 17:04:00.209517: I tensorflow_serving/model_servers/server_core.cc:514]  (Re-)adding model: deka
        2018-08-08 17:04:00.328483: I tensorflow_serving/core/basic_manager.cc:716] Successfully reserved resources to load servable {name: deka version: 16}
        2018-08-08 17:04:00.328529: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: deka version: 16}
        2018-08-08 17:04:00.328551: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: deka version: 16}
        2018-08-08 17:04:00.328611: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /home/deka/Desktop/test_tensorflow_serving/test_serving_model3/16
        2018-08-08 17:04:00.328649: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:242] Loading SavedModel with tags: { serve }; from: /home/deka/Desktop/test_tensorflow_serving/test_serving_model3/16
        2018-08-08 17:04:00.477419: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:161] Restoring SavedModel bundle.
        2018-08-08 17:04:00.675467: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:196] Running LegacyInitOp on SavedModel bundle.
        2018-08-08 17:04:00.775900: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:291] SavedModel load for tags { serve }; Status: success. Took 447238 microseconds.
        2018-08-08 17:04:00.776040: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /home/deka/Desktop/test_tensorflow_serving/test_serving_model3/16/assets.extra/tf_serving_warmup_requests
        2018-08-08 17:04:00.776328: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: deka version: 16}
        2018-08-08 17:04:00.838213: I tensorflow_serving/model_servers/main.cc:323] Running ModelServer at 0.0.0.0:6660 ...

<h2> Need test server. </h2>

        from grpc.beta import implementations # for request
        import tensorflow as tf
        import numpy
        import pandas

        from tensorflow.core.framework import types_pb2
        from tensorflow.python.platform import flags
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2


        tf.app.flags.DEFINE_string('server', 'localhost:6660',
                                   'inception_inference service host:port') # server port
        FLAGS = tf.app.flags.FLAGS


        def main(_):
            feed_value1 = float(input(" Input value to pred : ")) # input value
            feed_value2 = numpy.asarray(feed_value1) # convert to array
            # Prepare request
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'deka' # name your model. Need to be same, because when you run it you gave name for your model --model_name=deka
            
            request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
            request.inputs['inputs'].float_val.append(feed_value2)
            request.output_filter.append('outputs')
            # Send request
            host, port = FLAGS.server.split(':')
            channel = implementations.insecure_channel(host, int(port))
            stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
            prediction = stub.Predict(request, 5.0)  # 5 secs timeout
            predd = numpy.asarray(prediction)
            floats = prediction.outputs['outputs'].float_val # take prediction
            pred_array = numpy.asarray(floats) # convert prediction to numpy array
            df = pandas.DataFrame({"predicted_value":pred_array}) # convert numpy array to pandas DataFrame
            print(df) # Show it


        if __name__ == '__main__':
            tf.app.run()

<p>Run this to test server: deka@grave:~/Desktop/test_tensorflow_serving$ python lil_test_server01.py</p>

        Input value to pred : 88.234
           predicted_value
        0      1345.742554

