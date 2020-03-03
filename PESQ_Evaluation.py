import pandas as pd
import os
import librosa
from pypesq import pesq
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

pesq_df = pd.read_csv('PESQ_Eval_Result.csv', index_col = 'fileName')

noisyPath = 'noisySpeech'
cleanPath = 'clean'

for fileName in tqdm(pesq_df.index):
    noisyFile= os.path.join(noisyPath,fileName)
    cleanFile= os.path.join(cleanPath,fileName)
    
    clean, rate = librosa.load('clean/'+fileName, sr=8000) # downsampling wavfiles from 44100Hz to 8000Hz
    noisySpeech, rate = librosa.load('noisySpeech/'+fileName, sr=8000) # downsampling wavfiles from 44100Hz to 8000Hz

    score = pesq(clean, noisySpeech, rate)
    pesq_df.loc[fileName,'PESQ'] = score

pesq_df.to_csv('PESQ_Eval_Result.csv')