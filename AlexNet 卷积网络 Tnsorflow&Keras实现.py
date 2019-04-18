# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:04:53 2019

@author: yz250029
"""

#%% AlexNet 卷积网络模型

'''
共6亿3000万个连接，参数共6000万，神经元数量65万个，卷积层5个，池化层3个，全连接层3个，最后一个softmax单元数1000，用以完成1000分类
C1：  卷积层 对224*224*3的 图片使用 96个 11*11 进行滤波，stride 为4*4  = 96个，得到55*55特征图 然后Relu去线性
N1.5: LRN： local response normalization
S2:   max_pool 3*3, stride 2， 得到96*27*27

C3:   卷积层 对96*27*27*3 使用256个 5*5*3 卷积， stride 为1*1 得到256个27*27， relu 
C3.5  LRN
S4:   max_pool 3*3  stride 2  = 256*13*13

C5:   卷积层 没有池化和LRN操作， 使用3*3卷积核，stride 为1 共384个， relu后得到384*13*13
C6:   卷积层 使用256个3*3 stride 为1， 得到256*13*13 Relu
C7：  卷积层 使用256个 3*3 stride 为1 得到256*13*13 Relu
S8:   池化层 使用3*3 stride 为2 的池化，压平

full9： 全连接 2048个神经元(relu)
full10：全连接 2048个神经元(relu)
softmax: 全连接 1000个结果，对应1000分类


'''

#%% Tensorflow 实现

#由于训练真的耗时，我们使用假数据模拟耗费的时间。由于没有标注，我们没有定义loss

import tensorflow as tf
import math
import time
from datetime import datetime
batch_size  = 32
num_batches = 100

def inference_op(images):
    parameters = []
    #Conv1
    with tf.name_scope('conv1'):
        kernel = tf.Variable(tf.truncated_normal([11,11,3,96],dtype = tf.float32,stddev=1e-1),name='weights')
        conv   = tf.nn.conv2d(images,kernel,[1,4,4,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [96],dtype = tf.float32),trainable=True,name='biases')
        conv1  = tf.nn.relu(tf.nn.bias_add(conv,biases))
        
        #打印第一层结构
        print(conv1.op.name,'',conv1.get_shape().as_list())
        parameters += [kernel,biases]
    
    #LRN and Maxpool
    lrn1  = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name="pool1")
    print(pool1.op.name,'',pool1.get_shape().as_list())


    #conv2    
    with tf.name_scope('conv2'):
        kernel = tf.Variable(tf.truncated_normal([5,5,96,256],dtype = tf.float32,stddev=1e-1),name='weights')
        conv   = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [256],dtype = tf.float32),trainable=True,name='biases')
        conv2  = tf.nn.relu(tf.nn.bias_add(conv,biases))
        parameters += [kernel,biases]
        #打印第一层结构
        print(conv2.op.name,'',conv2.get_shape().as_list())

    
    #LRN and Maxpool
    lrn2  = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name="pool2")
    print(pool2.op.name,'',pool2.get_shape().as_list())

    with tf.name_scope('conv3'):
        kernel = tf.Variable(tf.truncated_normal([3,3,256,384],dtype = tf.float32,stddev=1e-1),name='weights')
        conv   = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [384],dtype = tf.float32),trainable=True,name='biases')
        conv3  = tf.nn.relu(tf.nn.bias_add(conv,biases))      
        parameters += [kernel,biases]
        print(conv3.op.name,'',conv3.get_shape().as_list())
        
        # C3没有池化层
    with tf.name_scope('conv4'):
        kernel = tf.Variable(tf.truncated_normal([3,3,384,384],dtype = tf.float32,stddev=1e-1),name='weights')
        conv   = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [384],dtype = tf.float32),trainable=True,name='biases')
        conv4  = tf.nn.relu(tf.nn.bias_add(conv,biases))      
        parameters += [kernel,biases]
        print(conv4.op.name,'',conv4.get_shape().as_list())       
        
    #C5
    with tf.name_scope('conv5'):
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype = tf.float32,stddev=1e-1),name='weights')
        conv   = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [256],dtype = tf.float32),trainable=True,name='biases')
        conv5  = tf.nn.relu(tf.nn.bias_add(conv,biases))      
        parameters += [kernel,biases]
        print(conv5.op.name,'',conv5.get_shape().as_list())   
    #maxpool
    pool3 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool3')
    print(pool3.op.name,'',pool3.get_shape().as_list())   
    
    #压平
    pool_shape = pool3.get_shape().as_list()
    nodes      = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped   = tf.reshape(pool3,[pool_shape[0],nodes])
    #全连接
    with tf.name_scope('fc_1'):
        fc1_weights = tf.Variable(tf.truncated_normal([nodes,4096],dtype=tf.float32,stddev=1e-1),name='weights')
        fc1_bias    = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        fc_1        = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_bias)
        #fc_1 = tf.nn.dropout(fc_1,0.8) #dropout
        parameters += [fc1_weights,fc1_bias]
        print(fc_1.op.name,'',fc_1.get_shape().as_list()) 
        
    #fc2
    with tf.name_scope('fc_2'):
        fc2_weights = tf.Variable(tf.truncated_normal([4096,4096],dtype=tf.float32,stddev=1e-1),name='weights')
        fc2_bias    = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        fc_2        = tf.nn.relu(tf.matmul(fc_1,fc2_weights)+fc2_bias)
        #fc_2 = tf.nn.dropout(fc_2,0.8)   #dropout
        parameters += [fc2_weights,fc2_bias]
        print(fc_2.op.name,'',fc_2.get_shape().as_list()) 
        
    return fc_2, parameters


# 模拟正向传播
# 公司电脑垃圾显卡，不调用GPU了
    
#with tf.Graph().as_default():
#    # 创建模拟图片
#    image_size = 224
#    images     = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
#    
#    #foreward broadcasting
#    fc_2, parameters = inference_op(images)
#    init_op = tf.global_variables_initializer()
#    
#    #调用GPU最佳适配合并算法
#    config = tf.ConfigProto()
#    config.gpu_options.allocator_type = 'BFC'

with tf.Session() as sess:
    image_size = 224
    images     = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
    
    #foreward broadcasting
    fc_2, parameters = inference_op(images)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    num_steps_burn_in      = 10
    total_dura             = 0.0
    total_dura_squared     = 0.0
    back_total_dura        = 0.0
    back_total_dura_squared= 0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _          = sess.run(fc_2)
        duration   = time.time() - start_time
        if i >= num_steps_burn_in:
            if i %10 == 0:
                print('%s: step %d, duration = %.3f' %(datetime.now(),i-num_steps_burn_in,duration))
            total_dura += duration
            total_dura_squared+= duration * duration
    average_time = total_dura/num_batches
    print('%s: Forward across %d steps, %.3f +/- %.3f sec/batch' %(datetime.now(),num_batches,average_time,math.sqrt(total_dura_squared/num_batches-
                                                                   average_time*average_time)))
    
        
    # BP
    grad = tf.gradients(tf.nn.l2_loss(fc_2),parameters)
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = sess.run(grad)
        duration   = time.time() - start_time
        if i >= num_steps_burn_in:
            if i %10 == 0:
                print('%s: step %d, duration = %.3f' %(datetime.now(),i-num_steps_burn_in,duration))
                back_total_dura +=duration
                back_total_dura_squared  += duration * duration
    back_avg_t = back_total_dura / num_batches
    print('%s: Forward -backward across %d steps, %.3f +/- %.3f sec/batch' %(datetime.now(),num_batches,back_avg_t,math.sqrt(back_total_dura/num_batches-
                                                                   back_avg_t*back_avg_t)))
    

#%% Keras 实现

from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
#from keras.utils.np_utils import to_categorical  #有使用这个对y 添加label
 
from keras.initializers import RandomNormal
  
model = Sequential()  
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1, seed=None)))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1, seed=None)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1, seed=None)))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-1, seed=None)))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5)) 
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(1000,activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  

#加上fit语句即可训练
#男默女泪，mmp用keras真鸡儿方便建模