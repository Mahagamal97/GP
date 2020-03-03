# =======================================================================#
#  splitting clean speech into fixed intervals 10 seconds for each file  #
# =======================================================================#

import os
from scipy.io import wavfile
from tqdm import tqdm

def checkdir(cleandir):
    if os.path.exists(cleandir) is False:
        os.makedirs(cleandir)

root_path = '/media/hamza/4A128378128367B1/GP/cloned-repository/GP'


checkdir(os.path.join(root_path,'wav_dir')) # check if output dir exists

# # convert mp3 to wav
# podcasts = os.listdir(os.path.join(root_path, 'mp3_dir')) # list all mp3 files in mp3_dir
# for file in tqdm(podcasts):
#     name, ext = file.split('.')   # split fileName to name and extension
#     mp3_path = os.path.join(root_path,'mp3_dir', file)
#     wav_path = os.path.join(root_path,'wav_dir', name+'.wav')
  
#     if os.path.exists(wav_path) is False: # if file is not converted yet => covert it!
#         # os.system() lets you run any terminal commandline from python files
#         os.system('ffmpeg -i {} -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav {}'.format(mp3_path,wav_path) ) 
    
podcastNames = os.listdir(os.path.join(root_path, 'wav_dir')) # list all wav files in wav_dir
clean_path = os.path.join(os.path.join(root_path, 'clean'))
print(podcastNames)

for file, fileNames in tqdm(zip(clean_path, podcastNames)): # iterate over both filesPaths amd filesNames
    interval = 44100 * 10       # set intervals equal to 10 seconds
    #print("file = ",file,"...fileNames = ",fileNames)   
    
    wav_path = os.path.join(root_path, 'wav_dir', fileNames) # combine filepath with fileName
    rate, wav = wavfile.read(wav_path)
    wav = wav[rate*600:]        # skip the introduction of podcast ( first 10 minutes ) assuming that it's redundant 
    # wav.shape[0] = total number of samples
    stop = (wav.shape[0]//interval) * interval # to avoid having the last file as fractional number of mintues
    #print(wav_path)
    for i in tqdm(range (0, stop-interval , interval)):
        sample = wav[i:i+interval]
        cleandir = os.path.join (root_path,'clean')         # create clean directory
        checkdir(cleandir)
        fn, ext = fileNames.split('.')
        i = int(i/interval)   # as we iterate by interval period each iteration
        save_fn = str(os.path.join(cleandir, fn+'_{}.wav'.format(i)))
        if os.path.exists(save_fn) is True:
            continue
        wavfile.write(filename=save_fn, rate=rate, data=sample)
