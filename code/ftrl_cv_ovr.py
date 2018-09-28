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
from sklearn.metrics import roc_auc_score,log_loss,f1_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
from wordbatch.models import FTRL, FM_FTRL

import time
t1=time.time()
train = pd.read_csv('../input/train_set.csv')
test = pd.read_csv('../input/test_set.csv')
train_id = train[["id"]].copy()
test_id = test[["id"]].copy()

column="word_seg"
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

train_x=trn_term_doc.tocsr()
test_x=test_term_doc.tocsr()
y=(train["class"]-1).astype(int)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
model1 = SelectKBest(chi2, k=10000)
train_x=model1.fit_transform(train_x, y)
test_x=model1.transform(test_x)


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

        if clf_name=="lgb":
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

            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )

            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],class_num))
            pred=model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],class_num))
        if clf_name=="lr":
            model = LogisticRegression(C=4, dual=False)
            model.fit(tr_x, tr_y)
            pre = model.predict_proba(te_x)
            pred=model.predict_proba(test_x)

        if clf_name=="svm":
            model = svm.LinearSVC()
            model.fit(tr_x, tr_y)
            pre = model.decision_function(te_x)
            pred=model.decision_function(test_x)

        if clf_name=="ridge":
            model = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',
                        max_iter=100,   normalize=False, random_state=0,  tol=0.0025)
            model.fit(tr_x, tr_y)
            pre = model.predict(te_x)
            pred=model.predict(test_x)

        if clf_name=="roc":
            model = NearestCentroid()
            model.fit(tr_x, tr_y)
            pre = model.predict(te_x)
            pred=model.predict(test_x)

        if clf_name=="ftrl":
            model = FM_FTRL(
                    alpha=0.02, beta=0.01, L1=0.00001, L2=30.0,
                    D=tr_x.shape[1], alpha_fm=0.1,
                    L2_fm=0.5, init_fm=0.01, weight_fm= 50.0,
                    D_fm=200, e_noise=0.0, iters=3,
                    inv_link="identity",threads=15,
                )
            model.fit(tr_x, tr_y)
            pre = model.predict(te_x)
            pred=model.predict(test_x)

        train[test_index]=pre.reshape((-1,1))

        test_pre[i, :]= pred.reshape((-1,1))
        cv_scores.append(log_loss(te_y, pre))

        print("%s now score is:"%clf_name,cv_scores)
    test[:]=test_pre.mean(axis=0)
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,class_num),test.reshape(-1,class_num),np.mean(cv_scores)


def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lgb",1)
    return xgb_train, xgb_test,cv_scores

def lr(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lr",1)
    return xgb_train, xgb_test,cv_scores

def lsvc(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"svm",1)
    return xgb_train, xgb_test,cv_scores

def ridge(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"ridge",1)
    return xgb_train, xgb_test,cv_scores

def roc(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"roc",1)
    return xgb_train, xgb_test,cv_scores

def ftrl(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"ftrl",1)
    return xgb_train, xgb_test,cv_scores

for ovr in range(19):
    train_y=y.copy()
    train_y=(train_y==ovr).astype(int)
    import lightgbm
    from sklearn.cross_validation import KFold
    folds = 5
    seed = 2018


    kf = KFold(train_x.shape[0], n_folds=folds, shuffle=True, random_state=seed)
    lgb_train, lgb_test,m=ftrl(train_x, train_y, test_x)

    score=log_loss(train_y, lgb_train)
    score=str(score)[:7]
    print(score)
    #保存预测概率文件
    train_prob=pd.DataFrame(lgb_train)
    train_prob.columns=["svm_ovr_%s"%ovr]
    train_prob.to_csv('../sub_prob/train_prob_ftrl_cv_ovr_%s_%s.csv'%(ovr,score),index=None)
    #保存预测概率文件
    test_prob=pd.DataFrame(lgb_test)
    test_prob.columns=["svm_ovr_%s"%ovr]
    test_prob.to_csv('../sub_prob/test_prob_ftrl_cv_ovr_%s_%s.csv'%(ovr,score),index=None)


