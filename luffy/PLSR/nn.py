# Multilayer Perceptron
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import keras
import scipy.io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import os.path

"""
BOW_all use PCA.
"""
train_description_feature_map = pickle.load(open('./features/11_30_[train]_description_normalized_only_>1_vector_py27.pkl', 'rb'))
test_derscription_feature_map = pickle.load(open('./features/11_30_[test]_description_normalized_only_>1_vector_py27.pkl', 'rb'))

train_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], train_description_feature_map))
test_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], test_derscription_feature_map))

pca_model_name = "./models/pca_bow_all_to_512.pkl"
if os.path.exists(pca_model_name):
    pca = pickle.load(open(pca_model_name, 'rb'))
else:
    pca = PCA(n_components=512)
    pca.fit(train_bow_lst)
    pickle.dump(pca, open(pca_model_name, 'wb'), protocol=2)

des_BOW_all_train_pca = pca.transform(train_bow_lst)
des_BOW_all_test_pca = pca.transform(test_bow_lst)

print(des_BOW_all_test_pca.shape)

"""
Training data.
"""
train_img_map = pickle.load(open('./features/12_1_train_img_feature_map_py27.pkl', 'rb'))
test_img_map = pickle.load(open('./features/12_1_test_img_feature_map_py27.pkl', 'rb'))

train_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], train_img_map)))
test_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], test_img_map)))

"""
MLP: 

input: pool5:2048
output: BOW_all_pca: 512
"""
input_size = 2048
output_size = 512

model = Sequential()
model.add(Dropout(0.3))
model.add(Dense(input_size, 1600, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1600, 1200, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1200, 800, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(800, output_size, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy')

model_checkpoint = ModelCheckpoint('./models/' + 'weights_12_1_epoch_{epoch:02d}.h5', monitor='val_loss', save_best_only=True)

# model.fit(xt, yt, batch_size=64, nb_epoch=500, validation_data=(xs, ys), class_weight=W, verbose=0)

model.fit(train_pool5_img, des_BOW_all_train_pca, batch_size=64, nb_epoch=500, verbose=1, shuffle=True,
            validation_split=0.1,
            callbacks=[model_checkpoint])

preds = model.predict(test_pool5_img)

output_name = "./models/mlp_pool5_to_bow_all_pca_512.pkl"
pickle.dump(preds, open(output_name, 'wb'), protocol=2)

# preds[preds>=0.5] = 1
# preds[preds<0.5] = 0

# print f1_score(ys, preds, average='macro')