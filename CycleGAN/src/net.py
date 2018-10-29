# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:39:10 2018

@author: acer
"""

import tensorflow as tf
import tensorflow.contrib as tc
from utils import *

def leaky_relu(tensor,alpha=0.2):
    return tf.maximum(tf.minimum(0.0,alpha *tensor),tensor)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset
    
def discriminator(image,reuse=False,name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        net = leaky_relu(tc.layers.conv2d(image,64,4,2,padding='SAME', activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                          biases_initializer=None))
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,128,4,2,padding='SAME', activation_fn=None,
                                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                               biases_initializer=None)))
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,256,4,2,padding='SAME', activation_fn=None,
                                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                               biases_initializer=None)))
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,512,4,1,padding='SAME', activation_fn=None,
                                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                               biases_initializer=None)))
        net = tc.layers.conv2d(net,1,4,1,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)
        return net
    
def generator_unet(image,reuse=False,name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(image,64,4,2,padding='SAME',activation_fn = None,
                                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                               biases_initializer=None)))
        e1 = tf.identity(net)
        
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,128,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)))
        e2 = tf.identity(net)
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,256,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)))
        e3 = tf.identity(net)
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,512,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)))
        e4 = tf.identity(net)
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,512,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)))
        e5 = tf.identity(net)
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,512,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)))
        e6 = tf.identity(net)
        net = leaky_relu(tc.layers.batch_norm(tc.layers.conv2d(net,512,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None)))
        e7 = tf.identity(net)
        net = tc.layers.batch_norm(tc.layers.conv2d(net,512,4,2,padding='SAME',activation_fn = None,
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               biases_initializer=None))
        net = tf.nn.relu(net)
        
        net = tc.layers.conv2d_transpose(net,512,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e7],3)
        net = tc.layers.conv2d_transpose(net,512,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e6],3)
        net = tc.layers.conv2d_transpose(net,512,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e5],3)
        net = tc.layers.conv2d_transpose(net,512,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e4],3)
        net = tc.layers.conv2d_transpose(net,256,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e3],3)
        net = tc.layers.conv2d_transpose(net,128,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e2],3)
        net = tc.layers.conv2d_transpose(net,64,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        net = tc.layers.dropout(net,0.5)
        net = tf.concat([tc.layers.batch_norm(net),e1],3)
        net = tf.nn.relu(net)
        net = tc.layers.conv2d_transpose(net,3,4,2,padding='SAME', activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         biases_initializer=None)
        
        return tf.nn.tanh(net)
        
        
        