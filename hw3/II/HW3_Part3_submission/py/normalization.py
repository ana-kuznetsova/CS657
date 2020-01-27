######NORMALIZATION#########
import numpy as np 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
from speech_preproc import open_map_dicts

def standardize(vec, means, sds):
    return (vec - means)/sds


def normalize(vec, _min, _max):
    return (vec-_min)/(_max-_min)

def calc_norm_stats(paths_dict, transform, _mean, _std, _min, _max):
    '''
    This function calculates the statistics for modified (standardized/normalized)
    data. 
    Arguments:
        paths_dict: paths to mapping dictionaries
        transform: kind of transform applied to data 'stand' for standartization
        'norm' for normalization
        '_mean, _std, _min, _max': statistics calculated on unmodified data.
    Returns:
        Modified vectors
    '''
    counter = 0

    #Means
    freq_means = np.zeros(257)
    squared = np.zeros(257)
    num_segments = 0
    #STDs
    freq_sds = np.array([])
    #Minimum/max values
    mins = []
    maxs = []
    
    for k in tqdm(paths_dict.keys()):
        # make stfts for noisy files
        for n in paths_dict[k].split('|'):
            noisy_speech, sr = librosa.load(n, sr=None)
            stft_noisy = librosa.stft(noisy_speech, n_fft=512, hop_length=160, win_length=320)
            stft_noisy = 10*np.log10(np.abs(stft_noisy)+0.0000000001)

            if transform=='stand':
            #standardize stft matrices by freq
                stft_noisy = np.array([standardize(col, _mean, _std) for col in stft_noisy.T]).T
                freq_means+= np.sum(stft_noisy, axis=1)
                squared += np.sum(stft_noisy**2, axis = 1)
                num_segments+= stft_noisy.shape[1]
            elif transform=='norm':
                #normalize stft freqs
                stft_noisy = np.array([normalize(col, _min, _max) for col in stft_noisy.T]).T
                current_max = np.max(stft_noisy, axis=1)
                maxs.append(current_max)
                current_min = np.min(stft_noisy, axis=1)
                mins.append(current_min)
            else:
                print(transform + ' is not a valid transformation.')

    if transform=='stand':

        freq_means = freq_means/num_segments
        squared = squared/num_segments
        std_dev = np.sqrt(squared - (freq_means**2))
        return freq_means, std_dev

    elif transform=='norm':
        maxs = np.array(maxs)
        maxs = np.max(maxs, axis=0)
        mins = np.min(np.array(mins), axis=0)
        return maxs, mins
    


def plot_transformed(stat_vecs, out_path, y_label):
    '''
    Plots modified transformed data.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3))
    #convert to frequencies in Hz
    ax1.set_title('Train')
    ax1.plot(np.linspace(0, 8000, num=257), stat_vecs[0])
    ax2.set_title('Dev')
    ax2.plot(np.linspace(0, 8000, num=257), stat_vecs[1])

    for ax in (ax1, ax2):
        ax.set(xlabel='Frequency (Hz)', ylabel=y_label)

    plt.savefig(out_path+y_label+'.png', bbox_inches="tight")


'''

The data was stored as a mapping dictionary of paths where
the key was a clean speech file path and value is the mapping
of correspoding noisy files path.

DICT_GLOB variable is the path to dir where mapping dictionaries are stored.
'''
DICT_GLOB = '/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/mapping_dicts/'

dict_train = open_map_dicts(DICT_GLOB+'train.p')
dict_dev = open_map_dicts(DICT_GLOB+'dev.p')

#Load test stats
train_mean = np.load('stats/train_means.npy')
train_std = np.load('stats/train_std.npy')
train_min = np.load('stats/train_mins.npy')
train_max = np.load('stats/train_max.npy')


dev_mean = np.load('stats/dev_means.npy')
dev_std = np.load('stats/dev_std.npy')
dev_min = np.load('stats/dev_mins.npy')
dev_max = np.load('stats/dev_max.npy')


freq_means_train, stds_train = calc_norm_stats(dict_train, 'stand',\
                                                train_mean,\
                                                train_std,\
                                                train_min,\
                                                train_max)

freq_means_dev, stds_dev = calc_norm_stats(dict_dev, 'stand',\
                                                dev_mean,\
                                                dev_std,\
                                                dev_min,\
                                                dev_max)


max_train, min_train = calc_norm_stats(dict_train, 'norm',\
                                                train_mean,\
                                                train_std,\
                                                train_min,\
                                                train_max)

max_dev, min_dev = calc_norm_stats(dict_dev, 'norm',\
                                                dev_mean,\
                                                dev_std,\
                                                dev_min,\
                                                dev_max)

plot_transformed([freq_means_train, freq_means_dev], 'graphs/', 'Mean (standardized)')
plot_transformed([stds_train, stds_dev], 'graphs/', 'SD (standardized)')


plot_transformed([max_train, max_dev], 'graphs/', 'Max (normalized)')
plot_transformed([min_train, min_dev], 'graphs/', 'Min (normalized)')


