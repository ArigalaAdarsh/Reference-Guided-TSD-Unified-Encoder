#!/usr/bin/env python3
import argparse
import librosa
from tqdm import tqdm
import io
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import gzip
import h5py
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

####################################################### ConvNext Feature Extractor ##################################################
class ConvNext_conf():  # for all 32k sample rate models
    def __init__(self):
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 224
        self.fmin = 50
        self.fmax = 14000
        self.window = 'hann'
        self.center = True
        self.pad_mode = 'reflect'
        self.ref = 1.0
        self.amin = 1e-10
        self.top_db = None

config = ConvNext_conf() # change the configuration according to the feature extractor you need

spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, win_length=config.window_size, 
                                    window=config.window, center=config.center, pad_mode=config.pad_mode, freeze_parameters=True)

logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, fmin=config.fmin, 
                                    fmax=config.fmax, ref=config.ref, amin=config.amin, top_db=config.top_db, freeze_parameters=True)

def log_mel_spectrogram(x):
    x = spectrogram_extractor(x) 
    x = logmel_extractor(x).squeeze()
    return x

####################################################################################################################################


parser = argparse.ArgumentParser()
# parser.add_argument('input_csv')
# parser.add_argument('-o', '--output', type=str, required=True)
# parser.add_argument('-c', type=int, default=4)
parser.add_argument('-sr', type=int, default=32000) # change to 16000 as we are reading audio with sr = 16000
# parser.add_argument('-col',
#                     default='filename',
#                     type=str,
#                     help='Column to search for audio files')
# parser.add_argument('-cmn', default=False, action='store_true')
# parser.add_argument('-cvn', default=False, action='store_true')
# parser.add_argument('-winlen',
#                     default=40,
#                     type=float,
#                     help='FFT duration in ms')
# parser.add_argument('-hoplen',
#                     default=20,
#                     type=float,
#                     help='hop duration in ms')

parser.add_argument('-n_mels', default=64, type=int) # change to 64 as we are setting mel dim to 64
parser.add_argument('-audio_target_length', default=320000, type=int) # default to sr * 10 for 10 sec audio

ARGS = parser.parse_args()

what_type = 'train'  # we have to do it for train, test and validate all three one by on eby changing this param
weak_csv = '/home/ananth/Targeted_Audio_Search_Phase_2/data/flists/urban_sed_' + what_type +'_weak.tsv'
DF_weak = pd.read_csv(weak_csv, sep='\t',usecols=[0,1])  # only read first cols, allows to have messy csv

strong_csv = '/home/ananth/Targeted_Audio_Search_Phase_2/data/flists/urban_sed_' + what_type +'_strong_modified.tsv'

print('weak_csv ',weak_csv)
print('strong_csv ',strong_csv)
DF_strong = pd.read_csv(strong_csv,sep='\t',usecols=[0,1,2,3])

# MEL_ARGS = {
#     'n_mels': ARGS.n_mels,
#     'n_fft': 1024,
#     'hop_length': int(ARGS.sr * ARGS.hoplen / 1000),
#     'win_length': int(ARGS.sr * ARGS.winlen / 1000)
# }

# EPS = np.spacing(1)


def extract_feature(fname): # extract feature from the file but they are keeping sampling rate as 22050 while reading audio, original sr = 44100. Need to change so that they read audio with sr = 16000
    """extract_feature
    Extracts a log mel spectrogram feature from a filename, currently supports two filetypes:
    1. Wave
    2. Gzipped wave
    :param fname: filepath to the file to extract
    """
    ext = Path(fname).suffix
    try:
        if ext == '.gz':
            with gzip.open(fname, 'rb') as gzipped_wav:
                y, sr = sf.read(io.BytesIO(gzipped_wav.read()),
                                dtype='float32')
                # Multiple channels, reduce
                if y.ndim == 2:
                    y = y.mean(axis=0)
                y = librosa.resample(y, orig_sr = sr, target_sr = ARGS.sr)
        elif ext in ('.wav', '.flac'):
            # y, sr = sf.read(fname, dtype='float32')
            y, sr = librosa.load(fname, sr=ARGS.sr, mono=True)
            y = torch.tensor(y).reshape(1,-1)
            # if y.ndim > 1:
            #     y = y.mean(axis=0)
            # y = librosa.resample(y, orig_sr = sr, target_sr = ARGS.sr)


    except Exception as e:
        # Exception usually happens because some data has 6 channels , which librosa cant handle
        logging.error(e)
        logging.error(fname)
        raise
    
    # pad the signal to match target length
    if y.shape[1] < ARGS.audio_target_length:
        
        missing = max(ARGS.audio_target_length - y.shape[1], 0)
        y = torch.nn.functional.pad(y, (0, missing), mode="constant", value=0.0)
    
    elif y.shape[1] > ARGS.audio_target_length:
        y = y[:,:ARGS.audio_target_length]

    # compute melspec and return it
    # lms_feature = np.log(librosa.feature.melspectrogram(y=y, **MEL_ARGS) + EPS).T
    lms_feature = log_mel_spectrogram(y)
    return fname, lms_feature

frames_num = 1001 # 10 sec audio and convnext feature gives 1001 frames
num_freq_bin = 224 # convnext mels

h5_name = './ydc_data/urban_target_detection_' + what_type + '_32k_224_mels.h5'
# This is the meta file which stores the information about the file, target event, mel spectrogram and time of the event in the file

print(h5_name)
hf = h5py.File(h5_name, 'w')
hf.create_dataset(      # create a h5 file with 4 keys : [filename, target_event, mel_specgram, time(start, end)] 
    name='filename', 
    shape=(0,),
    maxshape=(None,),
    dtype='S80')
hf.create_dataset(
    name='target_event', 
    shape=(0,), 
    maxshape=(None,),
    dtype='S80')
hf.create_dataset(
    name='mel_feature', 
    shape=(0, frames_num, num_freq_bin), 
    maxshape=(None, frames_num, num_freq_bin), 
    dtype=np.float32)
hf.create_dataset(
    name='time',
    shape=(0,10,2),
    maxshape=(None,10,2),
    dtype=np.float32
)
weak_filename = DF_weak['filename']
weak_label = DF_weak['event_labels']
n=0
for i,filename in enumerate(weak_filename):
    basename = Path(filename).name
    print(i,basename)
    strong_name = '/home/ananth/datasets/URBAN-SED/audio/' + what_type + '/'+basename # validate/
    fname, lms_feature = extract_feature(strong_name)
    # print(fname,lms_feature.shape)
    # print(i,filename)
    event_labels = weak_label[i]
    ls_event = event_labels.split(',')
    new_ls_event = []

    ############ rejected list as we want it to adapt for open domain target sound detection ###########
    # rejected_ls = ['jackhammer','siren','street_music']

    for event in ls_event: # delete overlap item
        if event not in new_ls_event:
        # if event not in new_ls_event and event not in rejected_ls:
            new_ls_event.append(event)
    
    for event in new_ls_event: # for each event present in the file, this process will happen
        time_label = []
        for j,strong in enumerate(DF_strong['filename']):
            
            # print("Strong_name", strong_name)
            # print("Strong", strong)
            if strong_name == strong:
                
                if event == DF_strong['event_label'][j]:
                    st = DF_strong['onset'][j]
                    ed = DF_strong['offset'][j]
                    st = float(st)
                    ed = float(ed)
                    tmp = [st,ed]
                    time_label.append(tmp)
        hf['mel_feature'].resize((n + 1, frames_num, num_freq_bin))
        hf['mel_feature'][n] = lms_feature

        hf['filename'].resize((n+1,))
        hf['filename'][n] = basename.encode()

        hf['target_event'].resize((n+1,))
        hf['target_event'][n] = event.encode()

        hf['time'].resize((n+1,10,2))
        while len(time_label) < 10:  # Since in a file they have atmax 9 sound events happening. So for safety they created list of list where list is of len 10, and inside the list we are having [start,end] for that event and if that particular event occur multiple times in a list, we are storing multiple [start,end] in the list. 
            time_label.append([-1,-1])
        assert len(time_label) == 10
        time_label = np.array(time_label)
        hf['time'][n] = time_label
        n += 1
print(n)


