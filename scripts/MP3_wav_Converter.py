# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 02:40:45 2020

@author: Hussein
"""
import os
from tqdm import tqdm
from pydub import AudioSegment

root_path = '******' # Write your source path instead of *****
src_path=os.listdir(root_path)
fileNumber=1 # Just a counter for different names

for file in tqdm(src_path):
    sound = AudioSegment.from_mp3(os.path.join(root_path,file)) # Read the mp3 files
    sound.export("wav_{}.wav".format(fileNumber), format="wav") #Write the wav files
    fileNumber+=1