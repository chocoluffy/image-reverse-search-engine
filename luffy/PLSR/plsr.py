
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import os.path

"""
Stats:
- pca: 1024, plsr:600 score:0.35
- pca: 2048, plsr: 800 score
"""

# pca_param = 2048
plsr_param = 1000
# pca_model_name = "./models/pca_bow_all_to_2048.pkl"
model_name = "./models/pls_1000_pool5_to_BOW_all_pca_5816.pkl"


train_description_feature_map = pickle.load(open('./features/11_30_[train]_description_normalized_only_>1_vector_py27.pkl', 'rb'))
test_derscription_feature_map = pickle.load(open('./features/11_30_[test]_description_normalized_only_>1_vector_py27.pkl', 'rb'))

train_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], train_description_feature_map))
test_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], test_derscription_feature_map))


# if os.path.exists(pca_model_name):
#     pca = pickle.load(open(pca_model_name, 'rb'))
# else:
#     pca = PCA(n_components=pca_param) # PCA param!
#     pca.fit(train_bow_lst)
#     pickle.dump(pca, open(pca_model_name, 'wb'), protocol=2)

# des_BOW_all_train_pca = pca.transform(train_bow_lst)
# des_BOW_all_test_pca = pca.transform(test_bow_lst)
des_BOW_all_train_pca = train_bow_lst
des_BOW_all_test_pca = test_bow_lst
# print des_BOW_all_test_pca.shape

train_img_map = pickle.load(open('./features/12_1_train_img_feature_map_py27.pkl', 'rb'))
test_img_map = pickle.load(open('./features/12_1_test_img_feature_map_py27.pkl', 'rb'))

train_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], train_img_map)))
test_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], test_img_map)))

model = PLSRegression(n_components=plsr_param)
model.fit(train_pool5_img[:8000], des_BOW_all_train_pca[:8000])

pickle.dump(model, open(model_name, 'wb'), protocol=2)

print model.score(train_pool5_img[8000:], des_BOW_all_train_pca[8000:])