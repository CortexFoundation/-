#encoding=utf8
import pandas as pd
import lightgbm as lgb
import re
import time
import numpy as np
import math
import gc
import pickle
import os
from sklearn.metrics import roc_auc_score,log_loss
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

import time
t1=time.time()
train = pd.read_csv('../input/train_set.csv')
test = pd.read_csv('../input/test_set.csv')
test_id = pd.read_csv('../input/test_set.csv')[["id"]].copy()

column="word_seg"
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

train_x=trn_term_doc.tocsr()
test_x=test_term_doc.tocsr()
train_y=(train["classify"]-1).astype(int)
#################################

def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.zeros((folds,test_x.shape[0],class_num))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]

        train_matrix = clf.Dataset(tr_x, label=tr_y)
        test_matrix = clf.Dataset(te_x, label=te_y)

        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'multiclass',
                  'metric': 'multi_logloss',
                  'min_child_weight': 1.5,
                  'num_leaves': 2**5,
                  'lambda_l2': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.5,
                  'colsample_bylevel': 0.5,
                  'learning_rate': 0.1,
                  'scale_pos_weight': 20,
                  'seed': 2018,
                  'nthread': 16,
                  'num_class': class_num,
                  'silent': True,
                  }


        num_round = 2000
        early_stopping_rounds = 100
        if test_matrix:
            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )

            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],class_num))
            train[test_index]=pre
            test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],class_num))
            cv_scores.append(log_loss(te_y, pre))

        print("%s now score is:"%clf_name,cv_scores)
        break
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,class_num),test.reshape(-1,class_num),np.mean(cv_scores)


def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lgb",19)
    return xgb_train, xgb_test,cv_scores

import lightgbm
from sklearn.cross_validation import KFold
folds = 5
seed = 2018



kf = KFold(train_x.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test,m=lgb(train_x, train_y, test_x)

#保存概率文件
test_prob=pd.DataFrame(lgb_test)
test_prob.columns=["class_prob_%s"%i for i in range(1,lgb_test.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('../sub_prob/prob_lgb_baseline.csv',index=None)

#生成提交结果
preds=np.argmax(lgb_test,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('../sub/sub_lgb_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)

