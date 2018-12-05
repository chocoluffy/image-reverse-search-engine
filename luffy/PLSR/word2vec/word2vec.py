"""
Map each description to vector using word2vec.
"""

import os
import csv
import random
import gensim
import numpy as np

word2vec = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
print("Loaded word vectors successfully!")

def doc_to_vec(words, word2vec):
    # get list of word vectors in sentence
    word_vecs = [word2vec.get_vector(w) for w in words if w in word2vec.vocab]
    # return average
    return np.stack(word_vecs).mean(0)

import pickle
import os.path
from sklearn.decomposition import PCA

train_description_feature_map = pickle.load(open('./../features/12_4_[train]_description_feature_map_py27.pkl', 'rb'))
test_derscription_feature_map = pickle.load(open('./../features/12_4_[test]_description_feature_map_py27.pkl', 'rb'))
print "loaded description map..."

train_img_map = pickle.load(open('./../features/12_1_train_img_feature_map_py27.pkl', 'rb'))
test_img_map = pickle.load(open('./../features/12_1_test_img_feature_map_py27.pkl', 'rb'))
print "loaded img feature map..."

train_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], train_img_map)))
test_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], test_img_map)))

train_doc_vecs = np.asarray(list(map(lambda x: x["doc_vec"], train_description_feature_map)))
test_doc_vecs = np.asarray(list(map(lambda x: x["doc_vec"], test_derscription_feature_map)))

"""
PLSR
"""
from sklearn.cross_decomposition import PLSRegression
print "model training..."

model_name = "./../models/pls_800_pool5_to_word2vec_n_v_adj_300.pkl"
if os.path.exists(model_name):
    model = pickle.load(open(model_name, 'rb'))
else:
    # train PLSR.
    model = PLSRegression(n_components=800)
    model.fit(train_pool5_img, train_doc_vecs)
    pickle.dump(model, open(model_name, 'wb'), protocol=2)
print "model loaded..."