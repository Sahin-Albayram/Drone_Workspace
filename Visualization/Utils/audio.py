import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import librosa
from librosa import display
import sklearn
from IPython.display import Audio

def Audio_plot(Audio_file, sr,t):
    display.waveshow(Audio_file, sr=sr, alpha=0.4,offset=t, color='#A300F9',lw=3)

def Audio_plot_harm(Audio_harm,sr,t):
    display.waveshow(Audio_harm, sr=sr, alpha=0.3,offset=t, color='#FFB100',lw=2)

def normalize(x,axis=0):
    return sklearn.preprocessing.minmax_scale(x,axis=0)

def Audio_visualize(Audio_file, sample_rate, time_start, time_end):
    Audio_file_short = Audio_file[int(time_start*sample_rate):int(time_end*sample_rate)]
    Audio_file_time = len(Audio_file_short)/sample_rate
    print(Audio_file_time)
    
    Audio_file_harm , Audio_file_perc = librosa.effects.hpss(Audio_file_short)

    Audio_file_spectral = librosa.feature.spectral_centroid(Audio_file_short, sr= sample_rate)[0]
    frames = range(len(Audio_file_spectral))
    
    Audio_file_spectral_t = librosa.frames_to_time(frames)
    
    Audio_file_spectral_norm = normalize(Audio_file_spectral)

    Audio_file_rmse = librosa.feature.rms(Audio_file_short,frame_length=512, hop_length=256,center =True)[0]
    Audio_file_energy = np.array([
        sum(abs(Audio_file_short[i:i+512]**2))
        for i in range(0, len(Audio_file_short),256)
    ])
    frames = range(len(Audio_file_energy))
    Audio_file_energy_t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=256)
    Audio_file_energy_norm = normalize(Audio_file_energy)

    Audio_file_rolloff = librosa.feature.spectral_rolloff(y= Audio_file_short, sr= sample_rate, hop_length= 256, roll_percent= 0.95)
    Audio_file_rolloff_min = librosa.feature.spectral_rolloff(y= Audio_file_short, sr= sample_rate, hop_length= 256, roll_percent= 0.05)

    Audio_features = {
        'Audio': Audio_file_short,
        'harm': Audio_file_harm,
        'perc': Audio_file_perc,
        'spectral_centroid': Audio_file_spectral_norm,
        'spectral_time': Audio_file_spectral_t,
        'energy': Audio_file_energy_norm,
        'energy_time': Audio_file_energy_t
    }
    return Audio_features
