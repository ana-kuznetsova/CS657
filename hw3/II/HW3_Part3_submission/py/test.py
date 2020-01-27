import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
from keras.models import load_model

def standardize(vec, means, sds):
    means = means.reshape(means.shape[0],1)
    sds = sds.reshape(sds.shape[0],1)
    return (vec - means)/sds


def normalize(vec, _min, _max):
    _min = _min.reshape(_min.shape[0],1)
    _max = _max.reshape(_max.shape[0],1)
    return (vec-_min)/(_max-_min)


model = load_model('/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/models/normIRM.h5')

NOISY_PATH = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/IEEE/train_noisy/l40s10-1-3.wav'


test_mean = np.load('stats/stat_vecs/test_means.npy')
test_std = np.load('stats/stat_vecs/test_std.npy')
test_min = np.load('stats/stat_vecs/test_mins.npy')
test_max = np.load('stats/stat_vecs/test_max.npy')


noisy_speech_time, sr = librosa.load(NOISY_PATH,sr=None)
noisy_speech = 10*np.log10(np.abs(librosa.stft(noisy_speech_time, n_fft=512,hop_length=160,win_length=320)))
noisy_speech = normalize(noisy_speech, test_min, test_max)
noisy_phase = librosa.stft(noisy_speech_time, n_fft=512,hop_length=160,win_length=320)

print('Saving predicted....')
predicted = model.predict(noisy_speech.T)
librosa.output.write_wav('/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/models/normIRM_predicted.wav',
                          librosa.istft(predicted.T*noisy_speech + np.angle(noisy_phase)), sr, norm=False)

print('Saving figure...')

fig1 = plt.figure(figsize = (10,5))
plt.imshow(np.abs(predicted), aspect = "auto", origin="lowest", extent = [0, 311, 0, 8000])
plt.xlabel("No. of samples")
plt.ylabel("Frequency")
plt.title("normIRM")
plt.savefig('/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/models/norm_irm_predicted.png', bbox_inches="tight")
