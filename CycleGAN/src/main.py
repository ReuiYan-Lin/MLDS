# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:01:08 2018

@author: acer
"""

import tensorflow as tf
import os
from model import cycleGAN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset','monet2photo','path of the dataset')
tf.app.flags.DEFINE_integer('epoch',200,'# of epoch')
tf.app.flags.DEFINE_integer('epoch_step',100,'# of epoch to decay lr')
tf.app.flags.DEFINE_integer('batch_size',1,'# images in batch')
tf.app.flags.DEFINE_integer('train_size', 1000000,'# images used to train')
tf.app.flags.DEFINE_integer('load_size', 286,'scale images to this size')
tf.app.flags.DEFINE_integer('fine_size',256,'then crop to this size')
tf.app.flags.DEFINE_integer('ngf', 64,'# of gen filters in first conv layer')
tf.app.flags.DEFINE_integer('ndf', 64,'# of discri filters in first conv layer')
tf.app.flags.DEFINE_integer('input_nc',3,'# of input image channels')
tf.app.flags.DEFINE_integer('output_nc',3,'# of output image channels')
tf.app.flags.DEFINE_float('lr', 0.0002,'initial learning rate for adam')
tf.app.flags.DEFINE_float('beta1',0.5, 'momentum term of adam')
tf.app.flags.DEFINE_string('which_direction','AtoB','AtoB or BtoA')
tf.app.flags.DEFINE_string('phase', 'train', 'train, test')
tf.app.flags.DEFINE_integer('save_freq', 1000,'save a model every save_freq iterations')
tf.app.flags.DEFINE_integer('print_freq',100, 'print the debug information every print_freq iterations')
tf.app.flags.DEFINE_boolean('continue_train', False, 'if continue training, load the latest model: 1: true, 0: false')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoint', 'models are saved here')
tf.app.flags.DEFINE_string('sample_dir', '../sample','sample are saved here')
tf.app.flags.DEFINE_string('test_dir', '../test','test sample are saved here')
tf.app.flags.DEFINE_float('L1_lambda', 10.0,'weight on L1 term in objective')
tf.app.flags.DEFINE_boolean('use_resnet', True,'generation network using reidule block')
tf.app.flags.DEFINE_boolean('use_lsgan',True, 'gan loss defined in lsgan')
tf.app.flags.DEFINE_integer('max_size', 50, 'max size of image pool, 0 means do not use image pool')

def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement = True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config = tfconfig) as sess:
        model = cycleGAN(sess,FLAGS)
        if FLAGS.phase == 'train':
            print("Training ...")
            model.train(FLAGS)
        #else: 
            #model.test(FLAGS)

if __name__ == '__main__':
    tf.app.run()