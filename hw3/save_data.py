import pickle
import librosa
import numpy as np
from tqdm import tqdm

print('Stufff')


def open_map_dicts(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


print('WHY do u do this to me?')


def IBM(noisy, clean):
    snr =20*np.log10((clean/noisy)+0.00000000000001)
    mask = np.around(snr, 0)
    mask[np.isnan(mask)] = 1
    mask[mask > 1] = 1
    return mask



def IRM(noisy, clean):
    b = 0.5
    snr =20*np.log10((clean/noisy)+0.00000000001)
    mask = np.power(snr/(snr + 1), b)
    return mask


def FFT(noisy, clean):
    return np.clip(20*np.log10((clean/noisy)+0.00000001), 0, 1)


def standardize(vec, means, sds):
        return (vec - means)/sds


def normalize(vec, _min, _max):
    return (vec-_min)/(_max-_min)



def save_data(DICT, STATS, MAXLEN, TRANSFORM, PART):
    
    train_tups = []

    print('Reading dict...')
    for i in DICT:
        clean_file = i
        clean_file = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/'+'/'.join(clean_file.split('/')[1:])
        clean_speech, sr = librosa.load(clean_file,sr=None)

        ##Pad data
        clean_speech = np.pad(clean_speech, MAXLEN, mode='constant')

        stft_clean = librosa.stft(clean_speech, n_fft=512,hop_length=160,win_length=320)
        stft_clean = np.abs(stft_clean)

        print('Iterating through noisy samples...')

        for p in DICT[i].split('|'):
            p = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/' + '/'.join(p.split('/')[1:])
            noisy_speech, sr = librosa.load(p ,sr=None)
            noisy_speech = np.pad(noisy_speech, MAXLEN, mode='constant')

            stft_noisy = librosa.stft(noisy_speech, n_fft=512,hop_length=160,win_length=320)
            stft_noisy = 10*np.log10(np.abs(stft_noisy)+0.00000001)
            
            if TRANSFORM=='norm':
                stft_noisy = np.array([normalize(col, STATS[0], STATS[1]) for col in stft_noisy.T]).T
                for j in range(stft_clean.shape[1]):
                    train_tups.append((stft_noisy[:,j], stft_clean[:,j]))
            elif TRANSFORM=='stand':
                stft_noisy = np.array([standardize(col, STATS[2], STATS[3]) for col in stft_noisy.T]).T
                for j in range(stft_clean.shape[1]):
                    train_tups.append((stft_noisy[:,j], stft_clean[:,j]))
            elif TRANSFORM=='normIBM':
                stft_noisy = np.array([normalize(col, STATS[0], STATS[1]) for col in stft_noisy.T]).T
                mask = IBM(stft_noisy, stft_clean)
                for j in range(stft_clean.shape[1]):
                    train_tups.append((mask[:,j], stft_clean[:,j]))
            elif TRANSFORM=='standIBM':
                stft_noisy = np.array([standardize(col, STATS[2], STATS[3]) for col in stft_noisy.T]).T
                mask = IBM(stft_noisy, stft_clean)
                for j in range(stft_clean.shape[1]):
                    train_tups.append((mask[:,j], stft_clean[:,j]))
            elif TRANSFORM=='normIRM':
                stft_noisy = np.array([normalize(col, STATS[0], STATS[1]) for col in stft_noisy.T]).T
                mask = IRM(stft_noisy, stft_clean)
                for j in range(stft_clean.shape[1]):
                    train_tups.append((mask[:,j], stft_clean[:,j]))
            elif TRANSFORM=='standIRM':
                stft_noisy = np.array([standardize(col, STATS[2], STATS[3]) for col in stft_noisy.T]).T
                mask = IRM(stft_noisy, stft_clean)
                for j in range(stft_clean.shape[1]):
                    train_tups.append((mask[:,j], stft_clean[:,j]))
            
            elif TRANSFORM=='normFFT':
                stft_noisy = np.array([normalize(col, STATS[0], STATS[1]) for col in stft_noisy.T]).T
                mask = FFT(stft_noisy, stft_clean)
                for j in range(stft_clean.shape[1]):
                    train_tups.append((mask[:,j], stft_clean[:,j]))
            elif TRANSFORM=='standFFT':
                stft_noisy = np.array([standardize(col, STATS[2], STATS[3]) for col in stft_noisy.T]).T
                mask = FFT(stft_noisy, stft_clean)
                for j in range(stft_clean.shape[1]):
                    train_tups.append((mask[:,j], stft_clean[:,j]))                
            

    print('Saving data...')
    np.save('/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/data/'+TRANSFORM+'_'+PART+'.npy', np.array(train_tups, dtype='float32'))



#DICT_PATH = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/IEEE/mapping_dicts/test.p'
#DICT = open_map_dicts(DICT_PATH)
#save_data(DICT)


def find_maxlen(DICT):

    print('Reading dict...')

    maxlen = 0

    for i in tqdm(DICT):
        clean_speech, sr = librosa.load(i,sr=None)
        length = len(clean_speech)
        if length > maxlen:
            maxlen = length
            del(clean_speech)
        else:
            del(clean_speech)
            continue

        for p in DICT[i].split('|'):
            noisy_speech, sr = librosa.load(p ,sr=None)
            if len(noisy_speech)>maxlen:
                maxlen = len(noisy_speech)
                del(noisy_speech)
            else:
                del(noisy_speech)
                continue

    print("MAXLEN:", maxlen)


print('Staring...')
DICT_PATH1 = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/IEEE/mapping_dicts/train.p'
DICT_PATH2 = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/IEEE/mapping_dicts/dev.p'
DICT_PATH3 = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/IEEE/mapping_dicts/test.p'
DICT_T = open_map_dicts(DICT_PATH1)
DICT_D = open_map_dicts(DICT_PATH2)
DICT_TEST = open_map_dicts(DICT_PATH2)

#DICT_T.update(DICT_D)
#DICT_T.update(DICT_TEST)
#find_maxlen(DICT_T)

##########LOAD STATS#############
#Load stat vectors used in standartization and normalization

MAXLEN = 59355

STATS_PATH = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/stats/' #change if needed


print('Loading stats...')

dev_max = np.load(STATS_PATH + 'dev_max.npy')
dev_mean = np.load(STATS_PATH + 'dev_means.npy')
dev_min = np.load(STATS_PATH + 'dev_mins.npy')
dev_std = np.load(STATS_PATH + 'dev_std.npy')


train_max = np.load(STATS_PATH + 'train_max.npy')
train_mean = np.load(STATS_PATH + 'train_means.npy')
train_min = np.load(STATS_PATH + 'train_mins.npy')
train_std = np.load(STATS_PATH + 'train_std.npy')

STATS_T = [train_min, train_max, train_mean, train_std]
STATS_D = [dev_min, dev_max, dev_mean, dev_std]


#############MAKE DATA ####################

#Combinations for training set
print('Calculating norm...')
#save_data(DICT_T, STATS_T, MAXLEN, 'norm', 'train')
#save_data(DICT_D, STATS_D, MAXLEN, 'norm', 'dev')

print('Calculating stand...')
#save_data(DICT_T, STATS_T, MAXLEN, 'stand', 'train')
#save_data(DICT_D, STATS_D, MAXLEN, 'stand', 'dev')

print('Calculating normIbm...')
save_data(DICT_T, STATS_T, MAXLEN, 'normIBM', 'train')
save_data(DICT_D, STATS_D, MAXLEN, 'normIBM', 'dev')

print('Calculating stand IBM...')
save_data(DICT_T, STATS_T, MAXLEN, 'standIBM', 'train')
save_data(DICT_D, STATS_D, MAXLEN, 'standIBM', 'dev')

print('Calculating norm IRM...')
save_data(DICT_T, STATS_T, MAXLEN, 'normIRM', 'train')
save_data(DICT_D, STATS_D, MAXLEN, 'normIRM', 'dev')


print('Calculating stand IRM...')
save_data(DICT_T, STATS_T, MAXLEN, 'standIRM', 'train')
save_data(DICT_D, STATS_D, MAXLEN, 'standIRM', 'dev')

print('Calculating norm FFT...')
save_data(DICT_T, STATS_T, MAXLEN, 'normFFT', 'train')
save_data(DICT_D, STATS_D, MAXLEN, 'normFFT', 'dev')

print('Calculating stand FFT....')
save_data(DICT_T, STATS_T, MAXLEN, 'standFFT', 'train')
save_data(DICT_D, STATS_D, MAXLEN, 'standFFT', 'dev')
