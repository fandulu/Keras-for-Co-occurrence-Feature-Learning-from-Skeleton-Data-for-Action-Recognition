from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras.optimizers import rmsprop
import tensorflow as tf

def one_obj(frame_l=16, joint_n=15, joint_d=3):

    input_joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
    input_joints_diff = Input(name='joints_diff', shape=(frame_l, joint_n, joint_d))
    
    ##########branch 1##############
    x = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(input_joints)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filters = 16, kernel_size=(3,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Permute((1,3,2))(x)
    
    x = Conv2D(filters = 16, kernel_size=(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)   
    ##########branch 1##############
    
    ##########branch 2##############Temporal difference
    x_d = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(input_joints_diff)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    
    x_d = Conv2D(filters = 16, kernel_size=(3,1),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Permute((1,3,2))(x_d)
    
    x_d = Conv2D(filters = 16, kernel_size=(3,3),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    ##########branch 2##############
    
    x = concatenate([x,x_d],axis=-1)
    
    x = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
       
    x = Conv2D(filters = 64, kernel_size=(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x) 
    x = Dropout(0.1)(x)
      
    model = Model([input_joints,input_joints_diff],x)

    return model

def multi_obj(frame_l=16, joint_n=15, joint_d=3):
    inp_j_0 = Input(name='inp_j_0', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_0 = Input(name='inp_j_diff_0', shape=(frame_l, joint_n, joint_d))
    
    inp_j_1 = Input(name='inp_j_1', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_1 = Input(name='inp_j_diff_1', shape=(frame_l, joint_n, joint_d))
    
    single = one_obj()
    x_0 = single([inp_j_0,inp_j_diff_0])
    x_1 = single([inp_j_1,inp_j_diff_1])
      
    x = Maximum()([x_0,x_1])
    
    x = Flatten()(x)
    x = Dropout(0.1)(x)
     
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(8, activation='sigmoid')(x)
      
    model = Model([inp_j_0,inp_j_diff_0,inp_j_1,inp_j_diff_1],x)
    
    return model