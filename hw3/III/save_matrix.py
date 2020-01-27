import os
import numpy as np

def save_mat(path_read, path_save):
    files = os.listdir(path_read)
    large_mat = 0
    for i in range(len(files)):
        if i == 0:
            large_mat = np.load(files[i])
        else:
            current = np.load(files[i])
            np.stack([large_mat, current], axis=0)
    np.save(path_save)



path_save = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/IEEE/ind_matrices/train_large.npy'

path_read = '/N/u/anakuzne/Carbonate/dl_for_speech/\
        HW3_II/IEEE/npy/train_frame/'

save_mat(path_read, path_save)