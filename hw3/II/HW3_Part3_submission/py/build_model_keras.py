import tensorflow as tf
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np


def show_NN_stats(history, out_path, title):
    '''
    Plot loss and accuracy for train and validation
    across the epochs.
    '''
    print(history.history.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3))
	#Plot accuracy
    ax1.set_title(title)
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')

    #Plot loss
    ax2.set_title(title)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.legend(['train', 'validation'], loc='upper left')
    plt.savefig(out_path+title+'.png', bbox_inches="tight")


#####LIST ALL THE PATHS#####
trainPath = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/data/stand_train.npy'
valPath = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/data/stand_dev.npy'
graph_output = '/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/model_graphs/'


print('Reading train data...')
trainData = np.load(trainPath)
np.random.shuffle(trainData)

trainSource = np.array([i[0] for i in trainData])
trainTarget = np.array([i[1] for i in trainData])

print('Reading val data...')
valData = np.load(valPath)
np.random.shuffle(valData)

valSource = np.array([i[0] for i in valData])
valTarget = np.array([i[1] for i in valData])


print('Initialize model...')
model = Sequential()
model.add(Dense(1024, input_dim=257, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(257, activation='relu'))

model.summary()


model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

print("Training model...")
history = model.fit(trainSource, trainTarget,
                    validation_data=(valSource, valTarget),
                    batch_size=10000,
                    epochs=25)


show_NN_stats(history, graph_output, 'standardized')

#save model
model.save('/N/u/anakuzne/Carbonate/dl_for_speech/HW3_II/py/models/standardized.h5')
