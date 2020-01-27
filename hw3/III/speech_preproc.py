import numpy as np
import librosa
from tqdm import tqdm
import argparse
import pickle

def open_map_dicts(path):
        with open(path, 'rb') as fp:
                return pickle.load(fp)



def save_npy(paths_dict, train_frame_path):
    #were k is the path to clean speech file
    
    counter = 0
    
    for k in tqdm(paths_dict.keys()):
        k_upd = '/'.join(k.split('/')[1:])
        #make stft for the clean file
        clean_speech, sr = librosa.load(k_upd,sr=None)
        stft_clean = librosa.stft(clean_speech, n_fft=512,hop_length=160,win_length=320)
        stft_clean = 10*np.log10(np.abs(stft_clean))
        
        # make stfts for noisy files
        for n in paths_dict[k].split('|'):
            n = '/'.join(n.split('/')[1:])
            noisy_speech, sr = librosa.load(n, sr=None)
            stft_noisy = librosa.stft(noisy_speech, n_fft=512, hop_length=160, win_length=320)
            stft_noisy = 10*np.log10(np.abs(stft_noisy))
            
            for j in range(stft_clean.shape[1]):
                Xfile = train_frame_path + 'x' + str(counter) +'.npy'
                Mfile = train_frame_path + 'm' + str(counter) +'.npy'
                #X is the magnitude STFT for noisy speech, M is the magnitude STFT for clean speech
                np.save(Xfile, stft_noisy[:,j])
                np.save(Mfile,stft_clean[:,j])
                counter+=1

def main():
    if __name__=="__main__":

        parser = argparse.ArgumentParser(description='Process speech data')
        parser.add_argument('-t', '--train_path', help='train_frame_path', required=True)
        parser.add_argument('-d', '--dict', help='Mapping dict of clean and noisy files', required=True)

        args = parser.parse_args()
        
        train_frame_path = args.train_path

        passed_dict = open_map_dicts(args.dict)
        
        save_npy(passed_dict, train_frame_path)

main()