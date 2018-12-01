import numpy as np
import os
import re
import pickle
import pprint

train_data_feature_map = pickle.load(open('./data/train_data_feature_map.pkl', 'rb'))
BOW_vector_map = pickle.load(open('./data/TRAIN_BOW_vector_map_572.pkl', 'rb'))

# print(train_data_feature_map[0])

"""
Random Forest.
"""
import numpy as np
X = []
for t in train_data_feature_map:
    X.append(t["FC_vector"])
X = np.asarray(X)
print(X.shape)
y = []
for l in BOW_vector_map:
    y.append(l["BOW_n_vector"])
y = np.asarray(y)
print(y.shape)

from sklearn.ensemble import RandomForestClassifier
import pickle

clf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)
clf.fit(X, y)
pickle.dump(clf, open("random_forest.pkl", 'wb'))
print("random forest trained")