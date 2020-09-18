from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from modules import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial

conv = partial(slim.conv2d, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
dconv = partial(slim.conv2d_transpose, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
fc = partial(ops.flatten_fully_connected, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
max_pooling = partial(slim.max_pool2d, kernel_size=2, stride=2, \
                                                        padding='VALID')
batch_norm = partial(slim.batch_norm, \
          decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
ln = slim.layer_norm          


'''
========================================================================
DCGAN FOR MNIST, similar to WGANGP code
https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py
========================================================================
'''

'''
The encoder for MNIST (28x28 images)
'''

def encoder_dcgan_mnist(img, x_shape, z_dim=100, dim=64, \
                             kernel_size=5, stride=2, \
                             name = 'encoder', \
                             reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    print("DCGAN encoder setup ---")
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, dim, kernel_size, stride))
        y = conv_bn_relu(y, dim * 2, kernel_size, stride)
        y = conv_bn_relu(y, dim * 4, kernel_size, stride)
        logit = fc(y, z_dim)
        return logit

'''
The generator for MNIST (28x28 images)
'''
def generator_dcgan_mnist(z, x_shape, dim=64, \
                       kernel_size=5, stride=2, \
                       name = 'generator', \
                       reuse=True, training=True):
                           
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, \
                            activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, \
                                                biases_initializer=None)
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]  
    print("DCGAN generator setup---")
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 4)
        y = tf.reshape(y, [-1, 4, 4, dim * 4])
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)
        # process the feature map 8x8 to to 7x7
        y = tf.reshape(y, [-1, 8 * 8 * 2 * dim])
        y = relu(fc(y, 7 * 7 * 2 * dim))
        y = tf.reshape(y, [-1, 7, 7, 2 * dim])
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)
        y = dconv(y, x_shape[2], kernel_size, stride)
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)

'''
The discriminator for MNIST (28x28 images)
'''               
def discriminator_dcgan_mnist(img, x_shape, dim=64, \
                             kernel_size=5, stride=2, \
                             name='discriminator', \
                             reuse=True, training=True):
                                 
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, \
                           activation_fn=lrelu, biases_initializer=None)
    print("DCGAN discriminator setup---")
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit),\
               logit,\
               tf.reshape(feature,[tf.shape(img)[0], -1])


