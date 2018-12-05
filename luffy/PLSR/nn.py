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

import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

"""
BOW_all use PCA.
"""
train_description_feature_map = pickle.load(open('./features/12_4_[train]_description_feature_map_py27.pkl', 'rb'))
test_derscription_feature_map = pickle.load(open('./features/12_4_[test]_description_feature_map_py27.pkl', 'rb'))

description_train_vecs = list(map(lambda x: x["doc_vec"], train_description_feature_map))
description_test_vecs = list(map(lambda x: x["doc_vec"], test_derscription_feature_map))

description_train_vecs = np.asarray(description_train_vecs)
description_test_vecs = np.asarray(description_test_vecs)
print(description_train_vecs.shape)

# pca_model_name = "./models/pca_bow_all_to_512.pkl"
# if os.path.exists(pca_model_name):
#     pca = pickle.load(open(pca_model_name, 'rb'))
# else:
#     pca = PCA(n_components=512)
#     pca.fit(description_train_vecs)
#     pickle.dump(pca, open(pca_model_name, 'wb'), protocol=2)

# des_BOW_all_train_pca = pca.transform(description_train_vecs)
# des_BOW_all_test_pca = pca.transform(description_test_vecs)

# print(des_BOW_all_test_pca.shape)

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
output_size = description_train_vecs.shape[1]

model = Sequential()
model.add(Dense(input_dim=input_size, output_dim=2048, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(input_dim=2048, output_dim=1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(input_dim=1024, output_dim=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(input_dim=512, output_dim=512, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(input_dim=4096, output_dim=output_size, activation='sigmoid'))
model.add(Dense(input_dim=512, output_dim=output_size, activation='linear'))

# model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
model.compile(optimizer=Adam(), loss='mse')

model_checkpoint = ModelCheckpoint('./models/' + 'weights_12_5_epoch_{epoch:02d}.h5', monitor='val_loss', save_best_only=True)

# model.fit(xt, yt, batch_size=64, nb_epoch=500, validation_data=(xs, ys), class_weight=W, verbose=0)

model.fit(train_pool5_img, description_train_vecs, batch_size=64, nb_epoch=500, verbose=1, shuffle=True,
            validation_split=0.1,
            callbacks=[model_checkpoint])

preds = model.predict(test_pool5_img)

output_name = "./models/nn_pool5_to_word2vec_300.pkl"
pickle.dump(preds, open(output_name, 'wb'), protocol=2)

from scipy.spatial.distance import cdist
import csv

dist = cdist(description_test_vecs, preds, 'cosine')
print("description * images dist matrix, shape:", dist.shape)
sorted_id = np.argsort(dist) # dist: N_description * N_images dist matrix.

with open('./NN_pool5_to_word2vec_300.csv', 'w') as csvfile:
        # write csv header
        fieldnames = ['Descritpion_ID', "Top_20_Image_IDs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, row in enumerate(sorted_id):
            top_choices =  list(map(lambda x: str(x) + ".jpg", row[:20]))
            res = {}
            res['Descritpion_ID'] = str(i) + ".txt" # file name
            res['Top_20_Image_IDs'] = " ".join(top_choices)
            writer.writerow(res)

print("Writing Complete.")