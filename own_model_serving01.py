from __future__ import print_function
import os
import sys
import numpy
rng = numpy.random
import tensorflow as tf
import matplotlib.pyplot as plt
from linear_input_data import  *

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/deka/Desktop/test_tensorflow_serving/test_serving_model3', 'Working directory.')
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

    # info about our data we run it from linear_input_data, linear_input_data.info_data
    info_data()
    # Hyperparameters
    learning_rate = 0.01
    display_step = 50
    # Train model
    print('Training model...')
    sess = tf.InteractiveSession()
    x,y,train_x,train_y,test_x,test_y = input_data() # take data from linear_input_data, linear_input_data.input_data()
    # Trainin example, as requested (Issue #1)
    train_X = numpy.asarray(train_x)
    train_Y = numpy.asarray(train_y)
    n_samples = train_X.shape[0]
    # Trainin example, as requested (Issue #1)
    test_X = numpy.asarray(test_x)
    test_Y = numpy.asarray(test_y)
    # tf Graph Input
    X = tf.placeholder(dtype=tf.float32,name="X")
    Y = tf.placeholder(dtype=tf.float32,name="Y")
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
    # Test our model
    predict_test = sess.run(pred, feed_dict={X: test_X})
    plt.plot(predict_test,color='green') # Predicted line
    plt.ylabel('Parameter EGT')
    plt.plot(test_Y,color='blue') # Test line
    plt.show()

    export_path_base = sys.argv[-1]
    export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(16)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
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

    tensor_info_x = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
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

    builder.save()
    print("Done exporting!")

if __name__ == '__main__':
    tf.app.run()
