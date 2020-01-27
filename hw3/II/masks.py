import numpy as np 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
from speech_preproc import open_map_dicts



def plot_masks(stft_noisy, stft_clean, mask, title):
    fig = plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)


    ax1.imshow(stft_clean, aspect='auto', origin='lowest', extent = [0, 267,0, 8000])
    ax1.set(title='STFT Clean Speech', xlabel='Time segments', ylabel='Frequency samples [Hz]')

    ax2.imshow(stft_noisy, aspect='auto', origin='lowest', extent = [0, 267,0, 8000])
    ax2.set(title='STFT Noisy Speech', xlabel='Time segments', ylabel='Frequency samples [Hz]')

    ax3.imshow(mask, origin='lowest', aspect='auto',  extent =[0, 267,0, 8000])
    ax3.set(title=title, xlabel='Time segments', ylabel='Frequency samples [Hz]')

    fig.savefig(title+".png", bbox_inches="tight")


def IBM(noisy, clean):
    snr = clean/noisy
    mask = np.around(snr, 0)
    mask[np.isnan(mask)] = 1
    mask[mask > 1] = 1
    return mask

def IRM(noisy, clean, b):
    b = 0.5
    snr =clean/noisy
    mask = np.power(snr/(snr + 1), b)
    return mask


def FFT(noisy, clean):
    return np.clip((clean/noisy), 0, 1)


DICT_PATH = '/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/mapping_dicts/train.p'
DICT = open_map_dicts(DICT_PATH)



key = list(DICT.keys())[0]
clean_speech, sr = librosa.load(key,sr=None)
stft_clean = np.abs(librosa.stft(clean_speech, n_fft=512,hop_length=160,win_length=320))

n = DICT[key].split('|')[0]
noisy_speech, sr = librosa.load(n, sr=None)


stft_noisy = librosa.stft(noisy_speech, n_fft=512, hop_length=160, win_length=320)
stft_noisy = np.abs(stft_noisy)

mask1 = IBM(stft_noisy, stft_clean)
mask2 = IRM(stft_noisy, stft_clean, 0.5)
mask3 = FFT(stft_noisy, stft_clean)

plot_masks(stft_noisy, stft_clean, mask1, 'IBM')
plot_masks(stft_noisy, stft_clean, mask2, 'IRM')
plot_masks(stft_noisy, stft_clean, mask3, 'FFT')