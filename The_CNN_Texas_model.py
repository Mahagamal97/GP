# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:35:21 2020

@author: Marwan
"""

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import librosa
import pandas as pd
from collections import deque
from sklearn.preprocessing import LabelEncoder
import scipy
import wave 


with open('/content/clean50file.pickle', 'rb') as f:
     cleanOutput= pickle.load(f)
with open('/content/noisy50file.pickle', 'rb') as f:
     noisyInput,noisy_phase= pickle.load(f)

cleanOutput=np.array(cleanOutput)
cleanOutput=cleanOutput[0]
#--------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------The model--------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Reshape  #new
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model  ##new
from keras import backend as K
import tensorflow as tf
from keras import models
from keras import layers
from keras.models import model_from_json
from keras.applications import resnet50
from keras.models import load_model
from keras.utils import to_categorical
import keras.regularizers as regularizers
from keras.utils import to_categorical


noisyInput=noisyInput.reshape(int(noisyInput.shape[0]/9),9,155,1)
cleanOutput=cleanOutput.reshape(cleanOutput.shape[0],155)


#========================================================
input_layer=Input(shape=(9,155,1))
firstConv=Conv2D(129, kernel_size=(5, 1),strides=1,use_bias=True,activation='relu')(input_layer)
secondConv=Conv2D(43, kernel_size=(5, 1),strides=3,use_bias=True,activation='relu')(firstConv)
flat=Flatten()(secondConv)
fullyConnec_layer=Dense(1024, activation='relu')(flat)
output_layer=Dense(155, activation='linear')(fullyConnec_layer)
#out_reshape=Reshape((155,1))(output_layer)
#print(noisyInput.shape, cleanOutput.shape)
model=Model(inputs=[input_layer],outputs=[output_layer])

model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(noisyInput,cleanOutput, epochs=20,validation_split=0.2)
#--------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------reconstruct--------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

reconstructed=model.predict(noisyInput) 

reconstructed=reconstructed.reshape(reconstructed.shape[0]//626,626,155) 
noisy_phase=noisy_phase.reshape(noisy_phase.shape[0]//129,129,626) 
#reconstructed=cleanOutput
print(reconstructed.shape) 

for k in range (reconstructed.shape[0]):
    suma=[]
    for i in range(reconstructed.shape[1]):
        for j in range(129):
            reconstructed[k][i][j]=math.sqrt(math.exp(reconstructed[k][i][j]))    
        suma.append(reconstructed[k][i]) 
    suma=np.array(suma) 
    suma=suma.T 
    the_real_STFT=suma[:-26 :] 
    print('the_rere',the_real_STFT.shape,the_real_STFT) 
    the_rec_stft=the_real_STFT*noisy_phase[k] 

    the_rec_signal=librosa.istft(the_rec_stft,hop_length= 128) 
   # all_sounds[k]=the_rec_signal 
    scipy.io.wavfile.write('recSignal_{}.wav'.format(k), 8000, the_rec_signal)