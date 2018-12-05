import pickle
import numpy as np
import pandas as pd

import os.path
import xgboost as xgb

"""
BOW_all use PCA.
"""
train_description_feature_map = pickle.load(open('./features/11_30_[train]_description_normalized_only_>1_vector_py27.pkl', 'rb'))
test_derscription_feature_map = pickle.load(open('./features/11_30_[test]_description_normalized_only_>1_vector_py27.pkl', 'rb'))

train_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], train_description_feature_map))
test_bow_lst = list(map(lambda x: x["BOW_all_normalized_vector"], test_derscription_feature_map))

print "Loaded description data..."


"""
Training data.
"""
train_img_map = pickle.load(open('./features/12_1_train_img_feature_map_py27.pkl', 'rb'))
test_img_map = pickle.load(open('./features/12_1_test_img_feature_map_py27.pkl', 'rb'))

train_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], train_img_map)))
test_pool5_img = np.asarray(list(map(lambda x: x["POOL_vector"], test_img_map)))
print "Loaded image data..."

"""
XGBoost
"""
dtrain = xgb.DMatrix(train_pool5_img, train_bow_lst)
dtest = xgb.DMatrix(test_pool5_img)

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model.save_model('./models/xgboost_trees_520_depth_4')
preds = model.predict(dtest)


from scipy.spatial.distance import cdist
import csv

dist = cdist(test_bow_lst, preds, 'cosine')
print("description * images dist matrix, shape:", dist.shape)
sorted_id = np.argsort(dist) # dist: N_description * N_images dist matrix.

with open('./submission/xgboost_520_4_pool5_to_bow_all.csv', 'w') as csvfile:
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