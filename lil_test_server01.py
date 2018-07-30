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
    predd = numpy.asarray(prediction)
    floats = prediction.outputs['outputs'].float_val
    pred_array = numpy.asarray(floats)
    df = pandas.DataFrame({"predicted_value":pred_array})
    print(df)


if __name__ == '__main__':
    tf.app.run()
