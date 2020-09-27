# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm,
  tf_ms_ssim,
  tf_ssim,
  blur_2th
)

import time
import os
import matplotlib.pyplot as plt
import cv2 

import numpy as np
import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=72,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        self.images_near = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_near')
    with tf.name_scope('VI_input'):
        self.images_far = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_far')


    with tf.name_scope('input'):
        self.input_image_near =self.images_near
        self.input_image_far =self.images_far


    with tf.name_scope('fusion'): 
        self.fusion_image=self.fusion_model(self.input_image_near,self.input_image_far)

    with tf.name_scope('grad_bin'):
        self.Image_far_grad=tf.abs(gradient(self.images_far))
        self.Image_near_grad=tf.abs(gradient(self.images_near))
        self.Image_fused_grad=tf.abs(gradient(self.fusion_image))
        self.Image_far_weight=tf.abs(blur_2th(self.images_far))
        self.Image_near_weight=tf.abs(blur_2th(self.images_near))
        
        self.Image_far_score=tf.sign(self.Image_far_weight-tf.minimum(self.Image_far_weight,self.Image_near_weight))
        self.Image_near_score=1-self.Image_far_score

        self.Image_far_score_ave=tf.reduce_mean(tf.sign((self.Image_far_weight-tf.minimum(self.Image_far_weight,self.Image_near_weight))))        
        self.Image_near_score_ave= 1- self.Image_far_score_ave
 
                      
        self.Image_far_near_grad_bin=tf.maximum(self.Image_far_grad,self.Image_near_grad)
        self.Image_fused_grad_bin=self.Image_fused_grad
    


    with tf.name_scope('image'):
        tf.summary.image('input_near',tf.expand_dims(self.images_near[1,:,:,:],0))  
        tf.summary.image('input_far',tf.expand_dims(self.images_far[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0)) 
        tf.summary.image('Image_far_grad',tf.expand_dims(self.Image_far_grad[1,:,:,:],0)) 
        tf.summary.image('Image_near_grad',tf.expand_dims(self.Image_near_grad[1,:,:,:],0))
        tf.summary.image('Image_far_near_grad_bin',tf.expand_dims(self.Image_far_near_grad_bin[1,:,:,:],0))
        tf.summary.image('Image_fused_grad_bin',tf.expand_dims(self.Image_fused_grad_bin[1,:,:,:],0))
        


          
    with tf.name_scope('d_loss'):        
        pos=self.discriminator(self.Image_far_near_grad_bin,reuse=False)
        neg=self.discriminator(self.Image_fused_grad_bin,reuse=True,update_collection='NO_OPS')
        pos_loss=tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        neg_loss=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))

        self.d_loss=neg_loss+pos_loss

        tf.summary.scalar('pos', tf.reduce_mean(pos))
        tf.summary.scalar('neg', tf.reduce_mean(neg))
        tf.summary.scalar('loss_d',self.d_loss)
        
        
        
    with tf.name_scope('g_loss'):
        self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        tf.summary.scalar('g_loss_1',self.g_loss_1)
        
        self.g_loss_int=tf.reduce_mean(self.Image_far_score*tf.square(self.fusion_image-self.images_far))+tf.reduce_mean(self.Image_near_score*tf.square(self.fusion_image-self.images_near))     
        self.g_loss_grad=tf.reduce_mean(self.Image_far_score*tf.square(gradient(self.fusion_image)-gradient(self.images_far)))+tf.reduce_mean(self.Image_near_score*tf.square(gradient(self.fusion_image)-gradient(self.images_near)))        
        self.g_loss_2 = 5 * self.g_loss_grad + self.g_loss_int



        tf.summary.scalar('self.g_loss_int', self.g_loss_int)
        tf.summary.scalar('self.g_loss_grad', self.g_loss_grad)
        tf.summary.scalar('g_loss_2',self.g_loss_2)
        self.g_loss_total=self.g_loss_1+10*self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss_total)        
    self.saver = tf.train.Saver(max_to_keep=50)
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_near")
      input_setup(self.sess,config,"Train_far")
    else:
      nx_near, ny_near = input_setup(self.sess, config,"Test_near")
      nx_far,ny_far=input_setup(self.sess, config,"Test_far")

    if config.is_train:     
      data_dir_near = os.path.join('./{}'.format(config.checkpoint_dir), "Train_near","train.h5")
      data_dir_far = os.path.join('./{}'.format(config.checkpoint_dir), "Train_far","train.h5")
    else:
      data_dir_near = os.path.join('./{}'.format(config.checkpoint_dir),"Test_near", "test.h5")
      data_dir_far = os.path.join('./{}'.format(config.checkpoint_dir),"Test_far", "test.h5")

    train_data_near= read_data(data_dir_near)
    train_data_far= read_data(data_dir_far)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    print(self.d_vars)
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)

    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)

    self.summary_op = tf.summary.merge_all()

    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()



    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_near) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_near = train_data_near[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_far = train_data_far[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          for i in range(2):
            _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_near: batch_images_near, self.images_far: batch_images_far})

          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_near: batch_images_near, self.images_far: batch_images_far})

          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d, err_g))


        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_near: train_data_near, self.images_far: train_data_far})
      result=result*127.5+127.5
      result = merge(result, [nx_near, ny_near])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def fusion_model(self,img_near,img_far):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1",[16],initializer=tf.constant_initializer(0.0))
            conv1_near= tf.nn.conv2d(img_near, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv1_near = lrelu(conv1_near)   
        with tf.variable_scope('layer1_far'):
            weights=tf.get_variable("w1_far",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_far",[16],initializer=tf.constant_initializer(0.0))
            conv1_far= tf.nn.conv2d(img_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv1_far = lrelu(conv1_far)           
            

####################  Layer2  ###########################            
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2",[16],initializer=tf.constant_initializer(0.0))
            conv2_near= tf.nn.conv2d(conv1_near, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv2_near = lrelu(conv2_near)         
            
        with tf.variable_scope('layer2_far'):
            weights=tf.get_variable("w2_far",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_far",[16],initializer=tf.constant_initializer(0.0))
            conv2_far= tf.nn.conv2d(conv1_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv2_far = lrelu(conv2_far)            
            

        conv_2_midle =tf.concat([conv2_near,conv2_far],axis=-1)    
       
  
        with tf.variable_scope('layer2_3'):
            weights=tf.get_variable("w2_3",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_near= tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv2_3_near = lrelu(conv2_3_near)   
                    
                       
        with tf.variable_scope('layer2_3_far'):
            weights=tf.get_variable("w2_3_far",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_3_far",[16],initializer=tf.constant_initializer(0.0))
            conv2_3_far= tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv2_3_far = lrelu(conv2_3_far)       
                        
####################  Layer3  ###########################               
        conv_12_near=tf.concat([conv1_near,conv2_near,conv2_3_near],axis=-1)
        conv_12_far=tf.concat([conv1_far,conv2_far,conv2_3_far],axis=-1)        
            
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3",[16],initializer=tf.constant_initializer(0.0))
            conv3_near= tf.nn.conv2d(conv_12_near, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv3_near =lrelu(conv3_near)
        with tf.variable_scope('layer3_far'):
            weights=tf.get_variable("w3_far",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_far",[16],initializer=tf.constant_initializer(0.0))
            conv3_far= tf.nn.conv2d(conv_12_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv3_far = lrelu(conv3_far)
            

        conv_3_midle =tf.concat([conv3_near,conv3_far],axis=-1)    
       
  
        with tf.variable_scope('layer3_4'):
            weights=tf.get_variable("w3_4",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_near= tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv3_4_near = lrelu(conv3_4_near)   
                    
                       
        with tf.variable_scope('layer3_4_far'):
            weights=tf.get_variable("w3_4_far",[1,1,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_4_far",[16],initializer=tf.constant_initializer(0.0))
            conv3_4_far= tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv3_4_far = lrelu(conv3_4_far)  


            
####################  Layer4  ########################### 
        conv_123_near=tf.concat([conv1_near,conv2_near,conv3_near,conv3_4_near],axis=-1)
        conv_123_far=tf.concat([conv1_far,conv2_far,conv3_far,conv3_4_far],axis=-1)                   
            
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4",[16],initializer=tf.constant_initializer(0.0))
            conv4_near= tf.nn.conv2d(conv_123_near, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv4_near = lrelu(conv4_near)
        with tf.variable_scope('layer4_far'):
            weights=tf.get_variable("w4_far",[3,3,64,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_far",[16],initializer=tf.constant_initializer(0.0))
            conv4_far= tf.nn.conv2d(conv_123_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv4_far = lrelu(conv4_far)
            
 
        conv_near_far =tf.concat([conv1_near,conv1_far,conv2_near,conv2_far,conv3_near,conv3_far,conv4_near,conv4_far],axis=-1)
 
        
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5_near= tf.nn.conv2d(conv_near_far, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_near=tf.nn.tanh(conv5_near)
    return conv5_near
    
    
    
  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        print(img.shape)
        with tf.variable_scope('layer_1'):
            weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
            conv1_far=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1_far = lrelu(conv1_far)
        with tf.variable_scope('layer_2'):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_far= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_far, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_far = lrelu(conv2_far)
        with tf.variable_scope('layer_3'):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_far= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_far, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_far=lrelu(conv3_far)
        with tf.variable_scope('layer_4'):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_far= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_far, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_far=lrelu(conv4_far)
            [B,H,W,C]=conv4_far.get_shape().as_list()          
 
            conv4_far = tf.reshape(conv4_far,[self.batch_size,H*H*256])
        with tf.variable_scope('line_5'):
            weights=tf.get_variable("w_5",[H*H*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_far, weights) + bias
    return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
