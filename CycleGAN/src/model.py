# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:28:00 2018

@author: acer
"""

import tensorflow as tf
from glob import glob
import numpy as np
from utils import *
from net import *
import time
import os

class cycleGAN(object):
    def __init__(self,sess,FLAGS):
        self.sess = sess
        self.batch_size = FLAGS.batch_size
        self.image_size = FLAGS.fine_size
        self.input_c_dim = FLAGS.input_nc
        self.output_c_dim = FLAGS.output_nc
        self.L1_lambda = FLAGS.L1_lambda
        self.dataset = FLAGS.dataset
        
        self.discriminator = discriminator
        
        self.generator = generator_unet

        '''
        if FLAGS.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if FLAGS.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion
        '''
        self.build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(FLAGS.max_size)
    
    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.input_c_dim+self.output_c_dim],
                                        name = 'real_A_and_B_images') # [batch,width,height,3+3 (channel)]
        self.real_A = self.real_data[:,:,:,:self.input_c_dim] # real image A (0~3)
        self.real_B = self.real_data[:,:,:,self.input_c_dim:self.input_c_dim+self.output_c_dim] # real image B (4~6)
        
        self.fake_B = self.generator(self.real_A,False,name='generatorA2B')
        self.fake_A_ = self.generator(self.fake_B,False,name='generatorB2A')
        self.fake_A = self.generator(self.real_B,True, name='generatorB2A')
        self.fake_B_ = self.generator(self.fake_A,True, name='generatorA2B')
        
        self.DB_fake = self.discriminator(self.fake_B,reuse = False,name='discriminatorB')
        self.DA_fake = self.discriminator(self.fake_A,reuse = False ,name = 'discriminatorA')
        
        self.g_loss_a2b =  tf.reduce_mean((self.DB_fake - tf.ones_like(self.DB_fake))**2) \
                            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A - self.fake_A_)) \
                            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))
        
        self.g_loss_b2a = tf.reduce_mean((self.DA_fake - tf.ones_like(self.DA_fake))**2) \
                            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A - self.fake_A_)) \
                            + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))
        
        self.g_loss = tf.reduce_mean((self.DA_fake - tf.ones_like(self.DA_fake))**2) \
                    + tf.reduce_mean((self.DB_fake - tf.ones_like(self.DB_fake))**2) \
                    + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A - self.fake_A_)) \
                    + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))
                    
        self.fake_A_sample = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.input_c_dim],
                                            name = 'fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.output_c_dim],
                                            name = 'fake_B_sample')
        
        self.DA_real = self.discriminator(self.real_A,reuse= True,name = 'discriminatorA')
        self.DB_real = self.discriminator(self.real_B,reuse= True,name = 'discriminatorB')
        
        self.DA_fake_sample = self.discriminator(self.fake_A_sample,reuse = True, name = 'discriminatorA')
        self.DB_fake_sample = self.discriminator(self.fake_B_sample,reuse = True, name = 'discriminatorB')
        
        self.db_loss_real = tf.reduce_mean((self.DB_real - tf.ones_like(self.DB_real))**2)
        self.db_loss_fake = tf.reduce_mean((self.DB_fake_sample - tf.zeros_like(self.DB_fake_sample))**2)
        
        self.db_loss = (self.db_loss_real + self.db_loss_fake) /2
        
        self.da_loss_real = tf.reduce_mean((self.DA_real - tf.ones_like(self.DA_real))**2)
        self.da_loss_fake = tf.reduce_mean((self.DA_fake_sample - tf.zeros_like(self.DA_fake_sample))**2)
        
        self.da_loss = (self.da_loss_real + self.da_loss_fake) /2
        
        self.d_loss = self.da_loss + self.db_loss
        
        self.g_loss_a2b_sum  = tf.summary.scalar("g_loss_a2b",self.g_loss_a2b)
        self.g_loss_b2a_sum  = tf.summary.scalar("g_loss_b2a",self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss",self.g_loss)
        
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum ,self.g_loss_b2a_sum,self.g_loss_sum])
        
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)
        
    def train(self,FLAGS):
        self.lr = tf.placeholder(tf.float32,None,name = 'learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr,beta1=FLAGS.beta1).minimize(self.d_loss,var_list = self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr,beta1=FLAGS.beta1).minimize(self.g_loss,var_list = self.g_vars)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("../logs", self.sess.graph)
        
        counter = 1
        start_time = time.time()
        
        if FLAGS.continue_train :
            if self.load(FLAGS.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        for epoch in range(FLAGS.epoch):
            #print('../data/{}/*.*'.format(self.dataset + '/trainA'))
            #dataA = os.listdir('../data/{}'.format(self.dataset+'/trainA'))
            
            
            dataA = glob('../data/{}/*.*'.format(self.dataset + '/trainA'))
            #print(dataA)
            dataB = glob('../data/{}/*.*'.format(self.dataset + '/trainB'))
            
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idx = min(min(len(dataA), len(dataB)), FLAGS.train_size) // self.batch_size
            lr = FLAGS.lr if epoch < FLAGS.epoch_step else FLAGS.lr*(FLAGS.epoch-epoch)/(FLAGS.epoch-FLAGS.epoch_step)
            
            for idx in range(0,batch_idx):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, FLAGS.load_size, FLAGS.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                
                fake_A,fake_B,_,summary_str = self.sess.run(
                        [self.fake_A,self.fake_B,self.g_optim,self.g_sum],
                        feed_dict={self.real_data:batch_images,self.lr:lr})
                self.writer.add_summary(summary_str,counter)
                [fake_A,fake_B] = self.pool([fake_A,fake_B])
                
                _,summary_str = self.sess.run(
                        [self.d_optim,self.d_sum],
                        feed_dict = {self.real_data:batch_images,
                                     self.fake_A_sample:fake_A,
                                     self.fake_B_sample:fake_B,
                                     self.lr: lr})
                self.writer.add_summary(summary_str,counter)
                
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idx, time.time() - start_time)))
                
                if np.mod(counter, FLAGS.print_freq) == 1:
                    self.sample_model(FLAGS.sample_dir, epoch, idx)

                if np.mod(counter, FLAGS.save_freq) == 2:
                    self.save(FLAGS.checkpoint_dir, counter)
                        
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save(self,checkpoint_dir,step):
        model_name = 'cycleGAN.model'
        model_dir = "%s_%s" % (self.dataset, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)    
        
    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('../data/{}/*.*'.format(self.dataset + '/testA'))
        dataB = glob('../data/{}/*.*'.format(self.dataset + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)
        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    '{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    '{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        
def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('../data/{}/*.*'.format(self.dataset + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('../data/{}/*.*'.format(self.dataset + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
