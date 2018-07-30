<h1> Linear Regression </h1>

<h2> linear_input_data.py </h2>
  This file will feed our model.
 
 
    import pandas as pd
    import numpy
    from sklearn.model_selection import train_test_split

 I took data from dataset, I took 2 parameters for linear regression, it's N and EGT.

    def input_data():
        df = pd.read_csv('/home/deka/Desktop/datasets/lil_dataset.csv')
        df.drop('dateandtime',axis=1,inplace=True)
        df.drop('Status',axis=1,inplace=True)
        df.dropna(inplace=True)
        z = df.values[:, 0]
        v = df.values[:, 1]
        
   Devide on train and test data.
        
        x_tr, x_te, y_tr, y_te = train_test_split(z,v,test_size=0.25)
        train_X = numpy.asarray(x_tr)
        train_Y = numpy.asarray(y_tr)
        test_X = numpy.asarray(x_te)
        test_Y = numpy.asarray(y_te)
        return z, v, train_X, train_Y, test_X, test_Y

 
 <h2> linear_input_data.py </h2>
 
  
  I took this linear regression from [example](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py).
 Save model from this [code](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py).
  
  
    from __future__ import print_function

    import os
    import sys
    import numpy
    rng = numpy.random

    import tensorflow as tf

    from linear_input_data import input_data

    tf.app.flags.DEFINE_integer('training_iteration', 1000,
                                'number of training iterations.')
    tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
    tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
    FLAGS = tf.app.flags.FLAGS


    def main(_):
        if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
          print('Usage: mnist_export.py [--training_iteration=x] '
            '[--model_version=y] export_dir')
          sys.exit(-1)
        if FLAGS.training_iteration <= 0:
          print('Please specify a positive value for training iteration.')
          sys.exit(-1)
        if FLAGS.model_version <= 0:
          print('Please specify a positive value for version number.')
          sys.exit(-1)

        # Parameters
        learning_rate = 0.01
        display_step = 50
        # Train model
        print('Training model...')
        sess = tf.InteractiveSession()
        x,y,train_x,train_y,test_x,test_y=input_data()
        # Trainin example, as requested (Issue #1)
        train_X = numpy.asarray(train_x)
        train_Y = numpy.asarray(train_y)
        n_samples = train_X.shape[0]
        # Testing example, as requested (Issue #2)
        test_X = numpy.asarray(test_x)
        test_Y = numpy.asarray(test_y)
        # tf Graph Input
        X = tf.placeholder("float",name="X")
        Y = tf.placeholder("float",name="Y")
        # Set model weights
        W = tf.Variable(rng.randn(), name="weight")
        b = tf.Variable(rng.randn(), name="bias")
        sess.run(tf.global_variables_initializer())
        # Construct a linear model
        pred = tf.add(tf.multiply(X, W), b, name="prediction")
        # Mean squared error
        cost = tf.reduce_sum(tf.pow(pred-Y, 2),name="cost_func")/(2*n_samples)
        # Gradient descent
        #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        for epoch in range(FLAGS.training_iteration):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
                # Display logs per epoch step
                if (epoch+1) % display_step == 0:
                    cosst = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
            predict = sess.run(pred, feed_dict={X: train_X})
            print("Prediction :", predict)

        export_path_base = sys.argv[-1]
        export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(16)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Build the signature_def_map.
        regression_inputs = tf.saved_model.utils.build_tensor_info(X)
        regression_outputs_prediction = tf.saved_model.utils.build_tensor_info(pred)

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

        tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(cost)

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

        builder.save()
        print("Done exporting!")

    if __name__ == '__main__':
        tf.app.run()
        
We need run this script, like this:

    python lil_model_serving01.py /home/deka/Desktop/test_tensorflow_serving/

This script will create trained model file "saved_model.pb" and his variables.
        
We need run server.

    tensorflow_model_server --port=9000 --model_name=deka --model_base_path=/home/deka/Desktop/test_tensorflow_serving/test_serving_model1
    
We will test the server, in this [script]():

    # Copyright 2016 Google Inc. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    """Manual test client for tensorflow_model_server."""

    # This is a placeholder for a Google-internal import.

    from grpc.beta import implementations
    import tensorflow as tf

    from tensorflow.core.framework import types_pb2
    from tensorflow.python.platform import flags
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2


    tf.app.flags.DEFINE_string('server', 'localhost:9000',
                               'inception_inference service host:port')
    FLAGS = tf.app.flags.FLAGS


    def main(_):
      # Prepare request
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'deka'
      request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
      request.inputs['inputs'].float_val.append(88.123) # input value
      request.output_filter.append('outputs')
      # Send request
      host, port = FLAGS.server.split(':')
      channel = implementations.insecure_channel(host, int(port))
      stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
      print(stub.Predict(request, 5.0))  # 5 secs timeout


    if __name__ == '__main__':
        tf.app.run()

 Run the script lil_test_server.py
 
    python lil_test_server.py
    
 Output:
 
     outputs {
      key: "outputs"
      value {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1341.0352783203125
      }
    }
    model_spec {
      name: "deka"
      version {
        value: 16
      }
      signature_name: "serving_default"
    }

Our predicted value float_val:1341.03

We can take just predicted value.

    from grpc.beta import implementations
    import tensorflow as tf
    import numpy
    import pandas

    from tensorflow.core.framework import types_pb2
    from tensorflow.python.platform import flags
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2


    tf.app.flags.DEFINE_string('server', 'localhost:9000',
                               'inception_inference service host:port')
    FLAGS = tf.app.flags.FLAGS


    def main(_):
        feed_value1 = float(input(" Input value to pred : "))
        feed_value2 = numpy.asarray(feed_value1)
        # Prepare request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'deka'
        request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
        request.inputs['inputs'].float_val.append(feed_value2)
        request.output_filter.append('outputs')
        # Send request
        host, port = FLAGS.server.split(':')
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        prediction = stub.Predict(request, 5.0)  # 5 secs timeout
        floats = prediction.outputs['outputs'].float_val
        pred_array = numpy.asarray(floats)
        df = pandas.DataFrame({"predicted_value":pred_array})
        print(df)


    if __name__ == '__main__':
        tf.app.run()

Input value to pred : 90.3434
 
Our output is dataframe.
 Output:
 
       predicted_value
    0      1376.257568



 GG WP
