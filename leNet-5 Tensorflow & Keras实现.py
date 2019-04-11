# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:53:57 2019

@author: yz250029
"""

#%%  leNet-5

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets("C:/Users/YZ250029/Desktop/其他/CNNlearning/",one_hot = True)
batch_size    = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 10000
#1: C1-conv 6*5*5 S2-maxpool 6*2*2  C3-conv 16*5*5 S4 16*2*2 C5 120*5*5 F6:84*(1*1)
def hidden_layer(input_tensor,regularizer,avg_class,resuse):
    # C1 特征为6* （28 * 28）
    with tf.variable_scope('C1-conv',reuse=resuse):
        conv1_weights = tf.get_variable('weight',[5,5,1,6],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases  = tf.get_variable('bias',[6],initializer = tf.constant_initializer(0.0))
        conv1         = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1         = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        
    with tf.name_scope('S2-max_pool',):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    with tf.variable_scope('C3-conv',reuse=resuse):
        conv2_weights = tf.get_variable('weight',[5,5,6,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #注：上一层池化得到了6个特征
        conv2_biases  = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv2         = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2         = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    
    with tf.name_scope('S4-max_pool',):  
        pool2         = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#        shape         = pool2.get_shape().as_list()  #[0]是batch中数据个数，[1]为长，[2]为宽，[3]为深度
#        nodes         = shape[1] * shape[2] * shape[3]  #5*5*16
#        reshaped      = tf.reshape(pool2,[shape[0],nodes])  #化成全连接输入
#       
#        
        
    with tf.variable_scope('C5-conv',reuse=resuse):
        conv3_weights = tf.get_variable('weight',[4,4,16,120],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases  = tf.get_variable('bias',[120],initializer = tf.constant_initializer(0.0))
        conv3         = tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding='VALID')
        relu3         = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
        shape         = relu3.get_shape().as_list() 
        nodes         = shape[1] * shape[2] * shape[3]
        reshaped      = tf.reshape(relu3,[shape[0],nodes])
        
    with tf.variable_scope('layer5-full1',reuse=resuse):
        Full_connection1_weights = tf.get_variable('weight',[nodes,84],initializer = tf.truncated_normal_initializer(stddev=0.1))
        #add normalization
        tf.add_to_collection('losses',regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable("bias",[84],initializer=tf.constant_initializer(0.1))
        
        
        if avg_class == None:
            Full_1    = tf.nn.relu(tf.matmul(reshaped,Full_connection1_weights)+Full_connection1_biases)
        else:
            Full_1    = tf.nn.relu(tf.matmul(reshaped,avg_class.average(Full_connection1_weights))+avg_class.average(Full_connection1_biases))
        #dropout
        Full_1 = tf.nn.dropout(Full_1,0.8)
        
    with tf.variable_scope("layer6-full2",reuse = resuse):
        Full_connection2_weight =tf.get_variable('weight',[84,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses',regularizer(Full_connection2_weight))
        Full_connection2_baises =tf.get_variable('bias',[10],initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            result = tf.matmul(Full_1,Full_connection2_weight)+Full_connection2_baises
        else:
            result = tf.matmul(Full_1,avg_class.average(Full_connection2_weight))+avg_class.average(Full_connection2_baises)
            
    return result
#%%
        
            
x = tf.placeholder(tf.float32,[batch_size,28,28,1],name='x')
y_= tf.placeholder(tf.float32,[None,10],name='y')
regularizer =  tf.contrib.layers.l2_regularizer(0.0001)

y = hidden_layer(x,regularizer,avg_class=None,resuse=False)
training_step = tf.Variable(0,trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(0.99,training_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

average_y = hidden_layer(x,regularizer,avg_class=variable_averages,resuse=True)  

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
learning_rate = tf.train.exponential_decay(learning_rate,training_step,minst.train.num_examples/batch_size,learning_rate_decay,staircase=True)
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=training_step)

with tf.control_dependencies([training_step,variables_averages_op]):
    train_op           = tf.no_op(name='train')
    crorent_prediction = tf.equal(tf.arg_max(average_y,1),tf.argmax(y_ ,1))
    accuracy= tf.reduce_mean(tf.cast(crorent_prediction,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_steps):
        if i % 500 ==0:
            x_val,y_val = minst.validation.next_batch(batch_size)
            reshaped_x2 = np.reshape(x_val,(batch_size,28,28,1))
            validate_feed = {x: reshaped_x2,y_:y_val}
            validate_accuracy = sess.run(accuracy,feed_dict = validate_feed)
            
            print('after %d steps, validation accuracy using average model is %g%%' %(i,validate_accuracy*100))
        x_train,y_train = minst.train.next_batch(batch_size) 
        reshaped_xs = np.reshape(x_train,(batch_size,28,28,1))
        sess.run(train_op,feed_dict = {x:reshaped_xs,y_:y_train})
#%%  KERAS 实现
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#from tensorflow.examples.tutorials.mnist import input_data
#minst = input_data.read_data_sets("C:/Users/YZ250029/Desktop/其他/CNNlearning/",one_hot = True)
# x_train,y_train = minst.train


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
#%% keras model 结构实现

from keras.layers import Input, Dense
from keras.models import Model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#from tensorflow.examples.tutorials.mnist import input_data
#minst = input_data.read_data_sets("C:/Users/YZ250029/Desktop/其他/CNNlearning/",one_hot = True)
# x_train,y_train = minst.train


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

inputs = Input(shape=(28,28,1))
x = Conv2D(6,kernel_size=(5,5), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(16,kernel_size=(5,5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(120,activation = 'relu')(x)
x = Dense(84,activation  = 'relu')(x)
predictions = Dense(10,activation  = 'softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))


