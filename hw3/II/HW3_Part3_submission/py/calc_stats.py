import numpy as np
import librosa
from tqdm import tqdm
import argparse
import pickle
import matplotlib.pyplot as plt


def plot_freqs(stat_vecs, out_path, y_label):
    '''
    The function plots statistics for the data set
    Argument:
        stat_vecs: mean, std vecs
        out_path: output directory where the graphs will be stored
        y_label: label for stats being plotted
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

def open_map_dicts(path):
        '''
        Loads pickled dicts
        '''
        with open(path, 'rb') as fp:
                return pickle.load(fp)

def calc_stats(paths_dict, prefix):
    #were k is the path to clean speech file   

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
        # where k is the path to clean speech
        #make stft for the clean file
        clean_speech, sr = librosa.load('/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/'+ k,sr=None)
        stft_clean = librosa.stft(clean_speech, n_fft=512,hop_length=160,win_length=320)
        stft_clean = np.abs(stft_clean)
        
        
        # make stfts for noisy files
        for n in paths_dict[k].split('|'):
            noisy_speech, sr = librosa.load('/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/'+n, sr=None)
            stft_noisy = librosa.stft(noisy_speech, n_fft=512, hop_length=160, win_length=320)
            stft_noisy = 10*np.log10(np.abs(stft_noisy)+0.0000000001)

            freq_means+= np.sum(stft_noisy, axis=1)
            squared += np.sum(stft_noisy**2, axis = 1)
            num_segments+= stft_noisy.shape[1]
            current_max = np.max(stft_noisy, axis=1)
            maxs.append(current_max)
            current_min = np.min(stft_noisy, axis=1)
            mins.append(current_min)

    # Calculate and save stats
    freq_means = freq_means/num_segments
    np.save('/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/HW3_Part3_submission/stat_vecs/'+prefix+'_means.npy', freq_means)
    squared = squared/num_segments
    std_dev = np.sqrt(squared - (freq_means**2))
    np.save('/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/HW3_Part3_submission/stat_vecs/'+prefix+'_std.npy', std_dev)
    maxs = np.array(maxs)
    maxs = np.max(maxs, axis=0)
    np.save('/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/HW3_Part3_submission/stat_vecs/'+prefix+'_max.npy', maxs)
    mins = np.min(np.array(mins), axis=0)
    np.save('/home/anakuz/data/docs/iu_courses/dl_for_speech/hw3/II/HW3_Part3_submission/stat_vecs/'+prefix+'_mins.npy', mins)

    return freq_means, std_dev, maxs, mins
    
def main():
    if __name__=="__main__":

        parser = argparse.ArgumentParser(description='Process speech data')
        parser.add_argument('-d', '--dict', help='Mapping dict of clean and noisy files', required=True)
        parser.add_argument('-o', '--out', help='Path for graph output', required=True)

        args = parser.parse_args()
        output = args.out

        passed_dict_train = open_map_dicts(args.dict+'train.p')
        passed_dict_dev = open_map_dicts(args.dict+'dev.p')
        passed_dict_test = open_map_dicts(args.dict+'test.p')
        print(passed_dict_test)
        
        freq_means_train, stds_train, max_train, min_train = calc_stats(passed_dict_train, 'train')
        freq_means_dev, stds_dev, max_dev, min_dev = calc_stats(passed_dict_dev, 'dev')
        freq_means_test, stds_test, max_test, min_test = calc_stats(passed_dict_test, 'test')

        plot_freqs([freq_means_train, freq_means_dev], output, 'Mean')
        plot_freqs([stds_train, stds_dev], output, 'Standard deviation')
        plot_freqs([max_train, max_dev], output, 'Maximum')
        plot_freqs([min_train, min_dev], output, 'Minimum')

main()
