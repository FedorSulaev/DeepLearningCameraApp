import tensorflow as tf
import numpy
import constants

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))
  
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding)
                        
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                            padding='SAME')

def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                            padding='SAME')
                            
def convolutional_layers():
    x = tf.placeholder(tf.float32, [None, None, None])
    
    # First layer
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_expanded = tf.expand_dims(x, 3)
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))
    
    # Second layer
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))
    
    # Third layer
    W_conv3 = weight_variable([5, 5, 128, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))
    
    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3]
                        
def get_training_model():
    x, conv_layer, conv_vars = convolutional_layers()
    
    # Densely connected layer
    W_fc1 = weight_variable([32 * 8 * 128, 2048])
    b_fc1 = bias_variable([2048])
    
    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
    
    # Output layer
    W_fc2 = weight_variable([2048, 1 + 7 * len(constants.CHARS)])
    b_fc2 = bias_variable([1 + 7 * len(constants.CHARS)])
    
    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])
    