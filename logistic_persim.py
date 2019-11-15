################################################################################
############### Spectral Clustering for Viewing Gesture Clusters ###############
############ Data set is persistence images - see pim_fullvec_set.py ###########
################################################################################

import numpy as np
import pandas as pd
import keras as ks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pim_df = pd.read_csv("./pim_vectors_mp40.csv")


pims = pim_df.values[:, :-2] # predictor vectors: persistence images (864x1600)
plt.matshow(pims[40, :].reshape(40, 40))
plt.show()
ks.utils.normalize(pims, axis=-1, order=2) # normalize persistence images
plt.matshow(pims[100, :].reshape(40, 40))
plt.show()

gests = pim_df.values[:, -2].astype("int64") # data labels: gesture numbers
unq_gests = np.unique(gests).size
# NOTE: gesture labels start at 1; keras must index categories from 0
gests = ks.utils.to_categorical(gests-1, 6)

#### train/ test split ####
np.random.seed(1)
pims_train, pims_test, gests_train, gests_test = train_test_split(
    pims,
    gests,
    test_size=0.2,
    random_state=1)


#### Computational Graph ####
batch = 1
inp_shape_x = (1600,)

log_reg = Sequential()

log_reg.add(Dense(
    input_shape=inp_shape_x,
    batch_size=batch,
    units=unq_gests,
    activation="softmax",
    use_bias=True))

print(log_reg.summary())

ks.utils.plot_model(
    log_reg,
    to_file='model.png',
    show_shapes=True)

# compile model w/ NAG optimizer: https://dominikschmidt.xyz/nesterov-momentum/
#nest = Nadam(lr = 0.005)
log_reg.compile(
    optimizer = "sgd", # switched to sgd for test
    loss = "categorical_crossentropy",
    metrics = ["accuracy"])

# train model
history = log_reg.fit(
    x=pims_train,
    y=gests_train,
    batch_size=batch,
    epochs=10,
    validation_split=0.2,
    verbose=True)

# plot fitting epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc='upper left')
plt.show()


score = log_reg.evaluate(pims_test, gests_test, verbose=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])
