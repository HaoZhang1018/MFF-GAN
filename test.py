# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import scipy.io as scio




def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img_near,img_far):
    with tf.variable_scope('fusion_model'):
    
####################  Layer1  ###########################
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_near= tf.nn.conv2d(img_near, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv1_near = lrelu(conv1_near)
        with tf.variable_scope('layer1_far'):
            weights=tf.get_variable("w1_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_far/w1_far')))
            bias=tf.get_variable("b1_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_far/b1_far')))
            conv1_far= tf.nn.conv2d(img_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv1_far = lrelu(conv1_far)    
                    



            
####################  Layer2  ###########################           
            
                      
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_near= tf.nn.conv2d(conv1_near, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_near = lrelu(conv2_near)
        with tf.variable_scope('layer2_far'):
            weights=tf.get_variable("w2_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_far/w2_far')))
            bias=tf.get_variable("b2_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_far/b2_far')))
            conv2_far= tf.nn.conv2d(conv1_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv2_far = lrelu(conv2_far)   
            
        conv_2_midle =tf.concat([conv2_near,conv2_far],axis=-1)      
        
        with tf.variable_scope('layer2_3'):
            weights=tf.get_variable("w2_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3/w2_3')))
            bias=tf.get_variable("b2_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3/b2_3')))
            conv2_3_near= tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv2_3_near = lrelu(conv2_3_near)
        with tf.variable_scope('layer2_3_far'):
            weights=tf.get_variable("w2_3_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3_far/w2_3_far')))
            bias=tf.get_variable("b2_3_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_3_far/b2_3_far')))
            conv2_3_far= tf.nn.conv2d(conv_2_midle, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_3_far = lrelu(conv2_3_far)               
            
            
                     
####################  Layer3  ###########################                 
        conv_12_near=tf.concat([conv1_near,conv2_near,conv2_3_near],axis=-1)
        conv_12_far=tf.concat([conv1_far,conv2_far,conv2_3_far],axis=-1)                   
         
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_near= tf.nn.conv2d(conv_12_near, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv3_near = lrelu(conv3_near)            
        with tf.variable_scope('layer3_far'):
            weights=tf.get_variable("w3_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_far/w3_far')))
            bias=tf.get_variable("b3_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_far/b3_far')))
            conv3_far= tf.nn.conv2d(conv_12_far, weights, strides=[1,1,1,1], padding='SAME')+ bias
            conv3_far =lrelu(conv3_far)            

        conv_3_midle =tf.concat([conv3_near,conv3_far],axis=-1)    
        
        with tf.variable_scope('layer3_4'):
            weights=tf.get_variable("w3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4/w3_4')))
            bias=tf.get_variable("b3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4/b3_4')))
            conv3_4_near= tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_4_near = lrelu(conv3_4_near)
        with tf.variable_scope('layer3_4_far'):
            weights=tf.get_variable("w3_4_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4_far/w3_4_far')))
            bias=tf.get_variable("b3_4_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_4_far/b3_4_far')))
            conv3_4_far= tf.nn.conv2d(conv_3_midle, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_4_far = lrelu(conv3_4_far)  



####################  Layer4  ###########################                 
        conv_123_near=tf.concat([conv1_near,conv2_near,conv3_near,conv3_4_near],axis=-1)
        conv_123_far=tf.concat([conv1_far,conv2_far,conv3_far,conv3_4_far],axis=-1)               
            
          
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_near= tf.nn.conv2d(conv_123_near, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_near = lrelu(conv4_near)
            
        with tf.variable_scope('layer4_far'):
            weights=tf.get_variable("w4_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_far/w4_far')))
            bias=tf.get_variable("b4_far",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_far/b4_far')))
            conv4_far= tf.nn.conv2d(conv_123_far, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_far = lrelu(conv4_far)            
            
        conv_near_far =tf.concat([conv1_near,conv1_far,conv2_near,conv2_far,conv3_near,conv3_far,conv4_near,conv4_far],axis=-1)
        
        
           
####################  Layer5  ###########################                          
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_near= tf.nn.conv2d(conv_near_far, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_near=tf.nn.tanh(conv5_near)

    return conv5_near



def input_setup(index):
    padding=0
    sub_near_sequence = []
    sub_far_sequence = []
    input_near=(imread(data_near[index])-127.5)/127.5
    input_near=np.lib.pad(input_near,((padding,padding),(padding,padding)),'edge')
    w,h=input_near.shape
    input_near=input_near.reshape([w,h,1])
    input_far=(imread(data_far[index])-127.5)/127.5
    input_far=np.lib.pad(input_far,((padding,padding),(padding,padding)),'edge')
    w,h=input_far.shape
    input_far=input_far.reshape([w,h,1])
    sub_near_sequence.append(input_near)
    sub_far_sequence.append(input_far)
    train_data_near= np.asarray(sub_near_sequence)
    train_data_far= np.asarray(sub_far_sequence)
    return train_data_near,train_data_far

for idx_num in range(19,20):
  num_epoch=idx_num
  while(num_epoch==idx_num):
      reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_60/CGAN.model-'+ str(num_epoch))

      with tf.name_scope('IR_input'):
          images_near = tf.placeholder(tf.float32, [1,None,None,None], name='images_near')
      with tf.name_scope('VI_input'):
          images_far = tf.placeholder(tf.float32, [1,None,None,None], name='images_far')
      with tf.name_scope('input'):
          input_image_near =images_near
          input_image_far =images_far

      with tf.name_scope('fusion'):
          fusion_image=fusion_model(input_image_near,input_image_far)

      with tf.Session() as sess:
          init_op=tf.global_variables_initializer()
          sess.run(init_op)
          data_near=prepare_data('Test_near')
          data_far=prepare_data('Test_far')
          for i in range(len(data_near)):
              start=time.time()
              train_data_near,train_data_far=input_setup(i)
              result =sess.run(fusion_image,feed_dict={images_near: train_data_near,images_far: train_data_far})
              print("result:",result.shape)
              result=result*127.5+127.5
              result = result.squeeze()
              image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
              if not os.path.exists(image_path):
                  os.makedirs(image_path)
              end=time.time()
              image_path = os.path.join(image_path,str(i+1)+".jpg")
              imsave(result, image_path)
              # scio.savemat(image_path, {'I':result})
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
      tf.reset_default_graph()
      num_epoch=num_epoch+1