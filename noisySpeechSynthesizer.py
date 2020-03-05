#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import librosa
from sklearn.utils import shuffle
from tqdm import tqdm
import os
import shutil
import scipy.io.wavfile

noise_df = pd.read_csv('noise_meta.csv', index_col = 'fileName')
clean_df = pd.read_csv('clean_meta.csv', names=["fileName"])


def checkdir(savedir):
    if os.path.exists(savedir) is False:
        os.makedirs(savedir)
    else:
        shutil.rmtree(savedir)   # remove a non empty directory
        os.makedirs(savedir)

def adjustLengths(clean_org,noise_org): # adjust lengths of clean and noise signals to be the same length so we can add them
    clean_len = len(clean_org)
    noise_len = len(noise_org)
    
    mxLength = max(clean_len, noise_len) # get the max length of noise and clean signals
    clean = np.empty(mxLength)
    noise = np.empty(mxLength)
    
    if clean_len < noise_len: # if noise signal > clean signal
        rep_time = int(np.floor(noise_len / clean_len)) # get the required reptition number for the signal
        left_len = noise_len - clean_len * rep_time     # get the rest decimal part because we used floor to calculate the repition number
        temp_data = np.tile(clean_org, [1, rep_time]) # repeat the clean signal with reptition number
        temp_data.shape = (temp_data.shape[1], )
        clean = np.hstack((temp_data, clean_org[:left_len])) # add the repeated clean signal with the rest decimal part as samples to get the same length as noise signal 
        noise = np.array(noise_org)
#         print("cleanShapeAdjusted in if=",clean.shape)
#         print("noiseShapeAdjusted in if=",noise.shape)

    else:
        rep_time = int(np.floor(clean_len / noise_len))
        left_len = clean_len - noise_len * rep_time
        temp_data = np.tile(noise_org, [1, rep_time])
        temp_data.shape = (temp_data.shape[1], )
        noise = np.hstack((temp_data, noise_org[:left_len]))
        clean = np.array(clean_org)
#         print("cleanShapeAdjusted =",clean.shape)
#         print("noiseShapeAdjusted =",noise.shape)
    
    return clean, noise


def SNRmixer(clean_org, noise_org, snr_dB): # adjust SNRs
    
    clean, noise = adjustLengths(clean_org, noise_org)
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5

    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5

    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr_dB/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return noisyspeech

dirpath = 'noisySpeech'

def noisySpeechGenerator(clean_df,noise_df,numNoisySpeech, numAddedNoises, snr): # generate noisy signals
    # clean_df = The clean dataframe
    # noise_df = The noise datafram
    # numNoisySpeech = the number of noisy signal you want to get ( ex. 5 means you will get 5 files with mixed clean and noise)
    # numAddedNoises = the number of noises you want to added for one clean signal ( let it 1 for now)
    checkdir(os.path.join(dirpath)) # check if output directory exists, if not create one

    shuffledClean = shuffle(clean_df)
    simpleClean = shuffledClean.iloc[:numNoisySpeech]
    simpleClean.set_index('fileName', inplace = True)
    #simpleClean.to_csv('simpleClean.csv')
    
    simpleNoise = pd.read_csv('simpleNoise.csv', index_col = 'fileName')
    simpleNoise.at[:,'length'] = 10.000125 
    pesq_csv = simpleClean
    
    fileCount=1
    for c, n in tqdm(zip(simpleClean.index, simpleNoise.index)):
        #totalNoise = np.empty(8000*int(simpleNoise['length'].mean())+1)  # htt3`yr 3lshan tnaseb ay 3add mn el addedNoises
        #rate, clean = scipy.io.wavfile.read('clean/'+c)
        clean, rate = librosa.load('clean/'+c, sr=8000) # downsampling wavfiles from 44100Hz to 8000Hz
        simpleClean.at[c,'length'] = clean.shape[0]/rate
        
        #i =0
        #for n in simpleNoise.index:
        noise, rate = librosa.load('noise_split/'+n, sr=8000) # downsampling wavfiles from 44100Hz to 8000Hz
        #rate, noise = scipy.io.wavfile.read('noise/'+n)
        #totalNoise += noise
        pesq_csv.loc[c,'added_noise'] = simpleNoise.loc[n,'label']       # assuming that we only add one noise type to each clean data
            #i += 1 
            #if (i > numAddedNoises):
            #    break
#         print("noisySpeechGenerator clean= ",clean.shape)
#         print("noisySpeechGenerator totalNoise= ",totalNoise.shape)
            
        noisySpeech  = SNRmixer(clean,noise,snr)
        
        #fn, ext = c.split('.')
        #noisyFile= os.path.join(dirpath, fn+'_{}.wav'.format(fileCount))
        noisyFile= os.path.join(dirpath, c)
        scipy.io.wavfile.write(noisyFile,rate, noisySpeech)
        
        pesq_csv.to_csv('PESQ_Eval_Result.csv')
        fileCount += 1
        
noisySpeechGenerator(clean_df,noise_df,5,1,0)
