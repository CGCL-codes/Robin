import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import h5py
import tensorflow as tf 
import time 
import sys
import os
import os.path
import pickle
 
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU,ELU,LeakyReLU
from keras.optimizers import SGD, Adam,RMSprop
from keras.engine.topology import Layer
from keras.engine import *
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets, preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.optimize import minimize_scalar
from scipy.io import loadmat

import classification.utils_load_learnsetup
from classification.LearnSetups.LearnSetup import LearnSetup
import lime.lime_tabular
from reader import reader_vector_simple_rand
from eval_methods import *


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

 
if K.backend() == "tensorflow":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    config = K.tensorflow_backend.tf.ConfigProto()
    config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = K.tensorflow_backend.tf.Session(config=config)
    K.set_session(session)


# Set parameters:
tf.set_random_seed(10086)
np.random.seed(10086)

maxlen = 800 
class_num = 204
#class_num = 70
batch_size = 204
#batch_size = 70
kernel_size = 3
epochs = 500
k = 10

RNN_denseneurons = 360
RNN_dropout = 0.6
RNN_l2reg = 1e-05
RNN_nounits = 288
#RNN_nounits = 256

###########################################
###############Load data###################
###########################################

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    return

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten



###########################################
###############Original Model##############
###########################################


def get_original_LearnSetup():
    """
    Build the original model to be explained. 

    """
    learnmodelspath = '/workspace/code-imitator-master/data/'
    feature_method: str = "CCS18"
    learn_method: str = "RNN"
    threshold_sel: int = 800
    PROBLEM_ID: str = "8294486_5681755159789568"
    #PROBLEM_ID: str = "1460488_1483485"
    testlearnsetup: LearnSetup = classification.utils_load_learnsetup.load_learnsetup(
        learnmodelspath=learnmodelspath,
        feature_method=feature_method,
        learn_method=learn_method,
        problem_id=PROBLEM_ID,
        threshold_sel=threshold_sel)
    return testlearnsetup

def load_data(originalLearnSetup, train_object, test_object, train = True): 
    """
    Load data.
    """
    x_train, x_test = train_object.getfeaturematrix(), test_object.getfeaturematrix()
    label_train, label_test = train_object.getlabels(), test_object.getlabels()
    
    if train:
        predclass_train = [originalLearnSetup.predict(feature_vec = x_train[i, :]) for i in range(0, x_train.shape[0])]
        predclass_test = [originalLearnSetup.predict(feature_vec = x_test[i, :]) for i in range(0, x_test.shape[0])]
        predclass_train = np.array(predclass_train)
        predclass_test = np.array(predclass_test)


        np.save('data/predclass_train.npy', predclass_train)
        np.save('data/predclass_test.npy', predclass_test)
    if not train:
        predclass_train = np.load('data/predclass_train.npy')
        predclass_test = np.load('data/predclass_test.npy')
    
    x_train = x_train.todense()
    x_test = x_test.todense()
    sc = preprocessing.StandardScaler().fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    return x_train, x_test, label_train, label_test, predclass_train, predclass_test

 

Invert = Lambda(lambda x: 1.-x)
Negative = Lambda(lambda x: -x)

def permute_dimensions(x):
    x1 = K.permute_dimensions(x, (0,2,1))
    return x1

def expand_dims(x):
    x1 = K.expand_dims(x, axis=-2)
    return x1

class Concatenate(Layer):
    """
    Layer for concatenation. 
    
    """
    def __init__(self, **kwargs): 
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs
        '''
        input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        '''
        return tf.concat([input1, input2], axis = -1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        # input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)

class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables. 

    """
    def __init__(self, tau0=0.5, k=k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
        # logits: [batch_size, d, 1]
        #logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]
        logits_ = logits
        d = int(logits_.get_shape()[2])
        unif_shape = [batch_size,self.k,d]

        uniform = K.random_uniform_variable(shape=unif_shape,
            low = np.finfo(tf.float32.as_numpy_dtype).tiny,
            high = 1.0)
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1) 
        logits = tf.reshape(logits,[-1, d]) 
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)

        output = K.in_train_phase(samples, discrete_logits) 
        return tf.expand_dims(output,-2)

    def compute_output_shape(self, input_shape):
        return input_shape


def construct_gumbel_selector(X_ph, y=None):


    if y is not None:
        hy = Dense(100)(y)
        hy = Dense(100, activation='relu')(hy)
        #hy = Dense(100, activation='relu')(hy)

    input_shape_lstm = (1, maxlen)
    
    #net = LSTM(RNN_nounits, return_sequences=True, input_shape=input_shape_lstm, dropout=RNN_dropout, name = 'lstm1_gumbel')(X_ph)
    #net = LSTM(RNN_nounits, return_sequences=True, dropout=RNN_dropout, name = 'lstm2_gumbel')(net)
    net = LSTM(RNN_nounits, return_sequences=False, input_shape=input_shape_lstm, dropout=RNN_dropout, name = 'lstm3_gumbel')(X_ph)
    combined = Concatenate()([hy, net])

    net = Dense(RNN_denseneurons, name='dense1_gumbel', kernel_regularizer=regularizers.l2(RNN_l2reg))(combined)
    net = Activation(activation='relu')(net)
    #net = Dense(round(RNN_denseneurons*0.8), name="dense2_gumbel", kernel_regularizer=regularizers.l2(RNN_l2reg))(net)
    #net = Activation(activation='relu')(net)
    logits_T = Dense(maxlen, name="dense3_gumbel", kernel_regularizer=regularizers.l2(RNN_l2reg))(net)
    logits_T = Lambda(expand_dims)(logits_T)
    
    return logits_T

def get_explainer(maxlen, k):
    X_ph = Input(shape=(1, maxlen), dtype='float32')
    y = Input(shape=(class_num,))

    #x = Lambda(permute_dimensions)(X_ph)
    logits_T = construct_gumbel_selector(X_ph, y=y) #[,800]
    
    tau = 0.5 
    T = Sample_Concrete(tau, k)(logits_T)
    #T = Lambda(permute_dimensions)(T)
    T_neg = Sample_Concrete(tau, k)(Negative(logits_T))
    #T_neg = Lambda(permute_dimensions)(T_neg)
    
    E = Model([X_ph,y], T)
    opt = Adam()  
    E.compile(loss='mse', optimizer=opt)   
    
    E_neg = Model([X_ph,y], T_neg)
    opt = Adam()  
    E_neg.compile(loss='mse', optimizer=opt)  
    
    E_pred = Model([X_ph,y], logits_T)
    opt = Adam()  
    E_pred.compile(loss='mse', optimizer=opt)  
     
    return E, T, E_neg, T_neg, E_pred

def get_approximator(maxlen):
    
    X_ph = Input(shape=(1, maxlen), dtype='float32')
    T = Input(shape=(1, maxlen))
    
    xmask = Multiply()([X_ph, T])
    x = BatchNormalization()(xmask, training=False)
    #x = LSTM(RNN_nounits, return_sequences=True, dropout=RNN_dropout)(x)
    #x = LSTM(RNN_nounits, return_sequences=True, dropout=RNN_dropout)(x)
    x = LSTM(RNN_nounits, return_sequences=False, dropout=RNN_dropout)(x)
    x = Dense(RNN_denseneurons, kernel_regularizer=regularizers.l2(RNN_l2reg))(x)
    x = Activation('relu')(x)
    #x = Dense(round(RNN_denseneurons*0.8), name="deep_representation", kernel_regularizer=regularizers.l2(RNN_l2reg))(x)
    #x = Activation('relu')(x)
    x = Dense(class_num, name="before_softmax", kernel_regularizer=regularizers.l2(RNN_l2reg))(x)
    preds = Activation('softmax')(x)
    
    A = Model([X_ph,T], preds)
    opt = Adam()  
    A.compile(loss='mse', optimizer=opt)   
       
    return A, preds

def get_model_pos(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_pred):

    set_trainability(E, False) 
    set_trainability(As, True) 
    set_trainability(Au, True) 

    T = E([X_ph,y])
    T_neg = Invert(T) 
    preds_s = As([X_ph,T])
    preds_u = Au([X_ph,T_neg])
     
    output_set = [preds_s,preds_u]
     
    model = Model(inputs=[X_ph,y],outputs=output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights, metrics=['accuracy'])
    
    return model,output_set

def get_model_neg(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_pred):

    set_trainability(E, True) 
    set_trainability(As, False) 
    set_trainability(Au, False) 

    T = E([X_ph,y])
    T_neg = Invert(T) 
    preds_s = As([X_ph,T])
    preds_u = Au([X_ph,T_neg])
    
    output_set = [preds_s,preds_u]
     
    model = Model(inputs=[X_ph,y],outputs=output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights, metrics=['accuracy'])
    
    return model,output_set
 
def build_model(maxlen, k):
  
    X_ph = Input(shape=(1, maxlen), dtype='float32')
    y = Input(shape=(class_num,))
    E,_,E_neg,_,E_pred=get_explainer(maxlen, k)   
    As,_=get_approximator(maxlen)
    Au,_=get_approximator(maxlen)
  
    opt = RMSprop()
  
    loss_weights = [1.,3e-2]
    loss = ['categorical_crossentropy','categorical_crossentropy']
    model_pos,_ = get_model_pos(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_pred)
 
    loss_weights = [1.,3e-2]
    loss = ['categorical_crossentropy','categorical_crossentropy']
    model_neg,_ = get_model_neg(X_ph,E,E_neg,As,Au,loss,opt,loss_weights,y,E_pred)
 
    return model_pos,model_neg,E,As,Au,E_pred


def robin(train = True): 

    flag_train_app = True

    #load original LearnSetup
    originalLearnSetup = get_original_LearnSetup()

    print('Loading dataset...') 
    train_object = originalLearnSetup.data_final_train
    test_object = originalLearnSetup.data_final_test

    
    with session as sess:
        ####TODO
        x_train, x_test, label_train, label_test, predclass_train, predclass_test = load_data(originalLearnSetup, train_object, test_object, train = False)
        train_acc = np.mean(predclass_train.tolist()==label_train)
        test_acc = np.mean(predclass_test.tolist()==label_test)
        print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, test_acc))

        print('Creating model...')
        model_pos,model_neg,E,As,Au,_ = build_model(maxlen, k)
        
        if train:
            y_train = to_categorical(predclass_train, class_num)
            y_test = to_categorical(predclass_test, class_num)
            data_reader_train = reader_vector_simple_rand.Reader(x_train, y_train, batch_size=batch_size,flag_shuffle=True,rng_seed=123)  
            bestf = 0.0
            bestuf = 0.0
            n_step = int(x_train.shape[0] * epochs / batch_size )
            
            step_list_sub = np.array([1,2,5]).astype(int) * 100
            step_list = []
            for i_ratio in range(20):
                step_list.extend(step_list_sub)
                step_list_sub = step_list_sub * 10
            step_list.append(n_step-1)
            
            i_step = -1
            while i_step < n_step:
                f = open(r'log/explainer_700.txt','a')
                i_step += 1
                
                if True:

                    x_batch, y_batch  = data_reader_train.iterate_batch()
                    if x_batch.shape[0]!=batch_size:
                        continue

                    y_batch_one_hot = to_categorical(np.argmax(y_batch, axis=-1), class_num)
                    
                    loss_pos=model_pos.train_on_batch([x_batch,y_batch_one_hot],[y_batch_one_hot,y_batch_one_hot])
                    loss_neg=model_neg.train_on_batch([x_batch,y_batch_one_hot],[y_batch_one_hot,1.-y_batch_one_hot])
                    #print(model_pos.metrics_names, file=f)
                    print("loss_pos: ", loss_pos, file=f)
                    #print(model_pos.metrics_names, file=f)
                    print("loss_neg: ", loss_neg, file=f)

                 
                #if i_step in step_list:
                if i_step%1 == 0:
                    print('------------------------ test ----------------------------', file=f)
                    
                    st = time.time()
                    duration = time.time() - st
                    print('TPS = {}'.format(duration/x_test.shape[0]), file=f)    
               
                    #fidelity,infidelity = eval_without_approximator(E, originalLearnSetup, 
                    #                                                x_test,y_test,flag_with_y=True,k=k,
                    #                                                )
                    #print('step: %d\tFS-M=%.4f\tFU-M=%.4f'%(i_step,fidelity,infidelity))
  
                    if flag_train_app:
                        y_test_one_hot = to_categorical(np.argmax(y_test, axis=-1), class_num)
                        mask = E.predict([x_test, y_test_one_hot], batch_size=1)
                        umask = np.ones_like(mask) - mask
                        sr = As.predict([x_test, mask], batch_size=1)
                        usr = Au.predict([x_test, umask], batch_size=1)
                        sr = np.argmax(sr, axis=-1)
                        usr = np.argmax(usr, axis=-1)
                        yclass = np.argmax(y_test_one_hot, axis=-1)
                        fidelity = np.mean(sr == yclass)
                        infidelity = np.mean(usr == yclass)
                        
                        bestf = fidelity
                        bestuf = infidelity
                        E.save('models/E.h5')
                        As.save('models/As.h5')
                        Au.save('models/Au.h5')
                        print('step: %d\tFS-A=%.4f\tFU-A=%.4f'%(i_step,fidelity,infidelity), file=f)
                f.close()                   
    
    return  

if __name__ == '__main__':
 

    import os,signal,traceback
    try:
        robin()
    except:
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)

    





