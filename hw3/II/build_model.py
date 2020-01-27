import pickle
import numpy as np
from tqdm import tqdm

import ffmpeg
import os
import torch
import torch.nn.functional as Func
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import copy



class trainDataLoader(data.Dataset):

    def __init__(self, file_path):
        self.trainData = np.load(file_path)
        self.data_arr = np.array([i[0] for i in self.trainData])
        self.label_arr = np.array([i[1] for i in self.trainData])

    def _getitem_(self, index):
        sample = torch.from_numpy(self.data_arr[index])
        enc_label = torch.from_numpy(self.laber_arr[index])

        return (sample, enc_label)

    def __len__(self):
        return len(self.data_arr)

class valDataLoader(data.Dataset):

    def __init__(self, file_path):
        self.valData = np.load(file_path)
        self.data_arr = np.array([i[0] for i in self.valData])
        self.label_arr = np.array([i[1] for i in self.valData] )

    def _getitem_(self, index):
        sample = torch.from_numpy(self.data_arr[index])
        enc_label = torch.from_numpy(self.laber_arr[index])

        return (sample, enc_label)

    def __len__(self):
        return len(self.data_arr)




trainPath = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/data/standIBM_train.npy'
valPath = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/data/standIBM_dev.npy'


#print('Loading train data...')
#trainData = np.load(trainData)
#np.random.shuffle(trainData)

#print('Loading val data...')
#valData = np.load(valData)
#np.random.shuffle(valData)



train = data.DataLoader(trainDataLoader(trainPath),batch_size = 10000, shuffle=True,drop_last = True) 
val = data.DataLoader(valDataLoader(valPath),batch_size = 10000, shuffle=True,drop_last = True) 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(257,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc4 = nn.Linear(1024,257)
        
    def forward(self,audio):
        audio = Func.relu(self.fc1(audio))
        audio = Func.relu(self.fc2(audio))
        audio = Func.relu(self.fc3(audio))
        audio = self.fc4(audio)
        return audio

def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias.data,0.1)


print("Initializng model...")
model = Net()
model.apply(weights)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()
model = model.to(device)
criterion.cuda()


num_epochs = 20
best_model = copy.deepcopy(model.state_dict())
best_loss = 9999

print('Start training...')
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    loss = 0.0 
    vali_loss = 0.0
    for step, (audio, target) in enumerate(train): 
        audio = audio.to(device)
        target = target.to(device)
        model.train()
        output = model(audio)
        newLoss = criterion(output,target)
        loss += newLoss.data
        #print(step,loss)
        optimizer.zero_grad()
        newLoss.backward()
        optimizer.step()
        print("Train step:"+str(step)+"/"+str(len(train)))
    for step, (audio, target) in enumerate(valN): 
        audio = audio.to(device)
        target = target.to(device)        
        model.eval()
        output = model(audio)
        new_valiLoss = criterion(output,target)
        vali_loss += new_valiLoss.data
        #print(step,vali_loss)
        if vali_loss < best_loss:
                best_loss = vali_loss
                best_model = copy.deepcopy(model.state_dict())
        print("Valid step:"+str(step)+"/"+str(len(valN)))
    print('Epoch:{:2},Loss:{:>.5f}'.format(epoch,loss))
    print('Epoch:{:2},Loss:{:>.5f}'.format(epoch,vali_loss)) 