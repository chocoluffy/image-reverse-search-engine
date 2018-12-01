
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import os.path

train_description_feature_map = pickle.load(open('./features/11_30_[train]_description_normalized_only_>1_vector_py27.pkl', 'rb'))
test_derscription_feature_map = pickle.load(open('./features/11_30_[test]_description_normalized_only_>1_vector_py27.pkl', 'rb'))

train_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], train_description_feature_map))
test_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], test_derscription_feature_map))

pca_model_name = "./models/pca_bow_all_to_3500.pkl"
if os.path.exists(pca_model_name):
    pca = pickle.load(open(pca_model_name, 'rb'))
else:
    pca = PCA(n_components=3500)
    pca.fit(train_bow_lst)
    pickle.dump(pca, open(pca_model_name, 'wb'), protocol=2)

des_BOW_all_train_pca = pca.transform(train_bow_lst)
des_BOW_all_test_pca = pca.transform(test_bow_lst)

print des_BOW_all_test_pca.shape

train_img_map = pickle.load(open('./features/12_1_train_img_feature_map_py27.pkl', 'rb'))
test_img_map = pickle.load(open('./features/12_1_test_img_feature_map_py27.pkl', 'rb'))

train_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], train_img_map)))
test_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], test_img_map)))

pls_pool5_to_BOW_all_pca_3500 = PLSRegression(n_components=800)
pls_pool5_to_BOW_all_pca_3500.fit(train_pool5_img[:8000], des_BOW_all_train_pca[:8000])
model_name = "./models/pls_pool5_to_BOW_all_pca_3500.pkl"
pickle.dump(pls_pool5_to_BOW_all_pca_3500, open(model_name, 'wb'), protocol=2)
print pls_pool5_to_BOW_all_pca_3500.score(train_pool5_img[8000:], des_BOW_all_train_pca[8000:])