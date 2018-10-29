import tensorflow as tf
import tensorflow.contrib as tc
import math
import tensorlayer.layers as tl
import numpy as np


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Generator(object):
    def __init__(self, 
        hidden_size, 
        img_row, 
        img_col):

        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        
    def __call__(self, seq_idx, z, reuse=False, train=True):

        batch_size = tf.shape(seq_idx)[0]

        tags_vectors = seq_idx

        with tf.variable_scope("g_net") as scope:
            if reuse:
                scope.reuse_variables()
            noise_vector = tf.concat([z,tags_vectors], axis=1)
            net_h0 = tc.layers.fully_connected(
                noise_vector, 64*12*12,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )
            net_h0 = tc.layers.batch_norm(net_h0,is_training=train)
            net_h0 = tf.reshape(net_h0,[-1,12,12,64])
            net = tf.nn.relu(net_h0)
            input_stage = net
            for i in range(1,17,1):
                name_scope = 'resblock_%d'%(i)
                net = residual_block(net, 64, 1, name_scope, train=train)
            net = tc.layers.batch_norm(net,is_training = train)
            net = tf.nn.relu(net)
            
            net = input_stage + net
            net = tc.layers.conv2d(net,256,3,1,padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None
                                   )
            net = pixelShuffler(net, scale=2)
            net = tc.layers.batch_norm(net,is_training = train)
            net = tf.nn.relu(net)
            
            net = tc.layers.conv2d(net,256,3,1,padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None
                                   )
            net = pixelShuffler(net, scale=2)
            net = tc.layers.batch_norm(net,is_training = train)
            net = tf.nn.relu(net)
            
            net = tc.layers.conv2d(net,256,3,1,padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None
                                   )
            net = pixelShuffler(net, scale=2)
            net = tc.layers.batch_norm(net,is_training = train)
            net = tf.nn.relu(net)
            
            net = tc.layers.conv2d(net,3,[9,9],[1,1],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None
                                   )
            net = tf.nn.tanh(net)
            print(net)
            return net
            
    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]
    
class Discriminator(object):
    def __init__(self, 
        hidden_size,
        img_row,
        img_col):

        self.hidden_size = hidden_size
        self.img_row = img_row
        self.img_col = img_col
        
    def __call__(self, seq_idx, img, reuse=True):

        batch_size = tf.shape(seq_idx)[0]

        tags_vectors = seq_idx
        print(img)
        with tf.variable_scope("d_net") as scope:
            if reuse == True:
                scope.reuse_variables()
            net = tc.layers.conv2d(img,32,[4,4],[2,2],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None 
                                   )
            net = leaky_relu(net)
            print(net)
            res = net
            
            net = discriminator_block(net,32,3,1,'disblock_1')
            
            net = discriminator_block(net,32,3,1,'disblock_1_1')
            
            net = tc.layers.conv2d(net,64,[4,4],[2,2],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None 
                                   )
            print(net)
            net = discriminator_block(net,64,3,1,'disblock_2_1')
            net = discriminator_block(net,64,3,1,'disblock_2_2')
            net = discriminator_block(net,64,3,1,'disblock_2_3')
            net = discriminator_block(net,64,3,1,'disblock_2_4')
            print(net)
            net = tc.layers.conv2d(net,128,[4,4],[2,2],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None 
                                   )
            net = leaky_relu(net)
            print(net)
            net = discriminator_block(net,128,3,1, 'disblock_3_1')
            net = discriminator_block(net,128,3,1, 'disblock_3_2')
            net = discriminator_block(net,128,3,1, 'disblock_3_3')
            net = discriminator_block(net,128,3,1, 'disblock_3_4')
            print(net)
            net = tc.layers.conv2d(net,256,[3,3],[2,2],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None 
                                   )
            net = leaky_relu(net)
            print(net)
            net = discriminator_block(net,256,3,1, 'disblock_4_1')
            net = discriminator_block(net,256,3,1, 'disblock_4_2')
            net = discriminator_block(net,256,3,1, 'disblock_4_3')
            net = discriminator_block(net,256,3,1, 'disblock_4_4')
            print(net)
            net = tc.layers.conv2d(net,512,[3,3],[2,2],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None 
                                   )
            net = leaky_relu(net)           
            print(net)
            net = discriminator_block(net,512,3,1, 'disblock_5_1')
            net = discriminator_block(net,512,3,1, 'disblock_5_2')
            net = discriminator_block(net,512,3,1, 'disblock_5_3')
            net = discriminator_block(net,512,3,1, 'disblock_5_4')
            print(net)
            net = tc.layers.conv2d(net,1024,[3,3],[2,2],padding = 'SAME',
                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation_fn=None 
                                   )
            net = leaky_relu(net) 
            print(net)
            net = tf.reshape(net,[-1,2*2*1024])
            print(net)
            with tf.variable_scope('dense_layer_1'):
                net_class = tf.layers.dense(net, 23, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
                print(net_class)
                net_class = tf.reshape(net_class,[-1,23])
                
            with tf.variable_scope('dense_layer_2'):
                net = tf.layers.dense(net, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

            return net,net_class
            
    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]
  
def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
    res = inputs

    with tf.variable_scope(scope):

        net = conv2d_sn(   inputs,  output_channel, kernel_size, kernel_size, stride, stride, stddev=0.02, name='conv1')
        #net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
        net = leaky_relu(net, 0.2)
        net = conv2d_sn(net,  output_channel, kernel_size, kernel_size, stride, stride, stddev=0.02, name='conv1')
        #net = conv2(net, kernel_size, output_channel, stride, use_bias=False, scope='conv2')
        net = net + res
        net = leaky_relu(net, 0.2)

    return net
def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def conv2d_sn(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    fan_in = k_h * k_w * input_.get_shape().as_list()[-1]

    fan_out = k_h * k_w * output_dim

    if stddev is None:
        stddev = np.sqrt(2. / (fan_in))

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv
	
def residual_block(tensor,outshape,stride,scope,train = True):
    net = tc.layers.conv2d(tensor,outshape,3,stride,padding = 'SAME',
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           activation_fn=None 
                           )  
    net = tc.layers.batch_norm(net,is_training = train)
    net = leaky_relu(net)
    net = tc.layers.conv2d(net,outshape,3,stride,padding = 'SAME',
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           activation_fn=None 
                           ) 
    net = tc.layers.batch_norm(net,is_training = train)
    net = net + tensor
    return net

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)