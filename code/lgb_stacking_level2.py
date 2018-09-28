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
from sklearn.linear_model import LogisticRegression

import time

t1=time.time()
train_lgb=pd.read_csv("../sub/train_prob_lgb.csv")
train_lr=pd.read_csv("../sub/train_prob_lr.csv")
train_rf=pd.read_csv("../sub/train_prob_rf.csv")
train_nn=pd.read_csv("../sub/train_prob_nn.csv")

test_lgb=pd.read_csv("../sub/test_prob_lgb.csv")
test_lr=pd.read_csv("../sub/test_prob_lr.csv")
test_rf=pd.read_csv("../sub/test_prob_rf.csv")
test_nn=pd.read_csv("../sub/test_prob_nn.csv")

train_y=(train_lgb["label"]).astype(int)

del train_lgb["label"],train_lr["label"],train_rf["label"]

train_x=np.concatenate([train_lgb.values,train_lr.values,train_rf.values,train_nn.values],axis=1)
test_x=np.concatenate([test_lgb.values,test_lr.values,test_rf.values,test_nn.values],axis=1)
#################################

def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.zeros((folds,test_x.shape[0],class_num))
    cv_scores=[]
    cv_scores_logloss=[]
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

            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,verbose_eval=50,
                              early_stopping_rounds=early_stopping_rounds
                              )

            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],class_num))
            pred=model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],class_num))
        if clf_name=="lr":
            model = LogisticRegression(C=4, dual=False)
            model.fit(tr_x, tr_y)
            pre = model.predict_proba(te_x)
            pred=model.predict_proba(test_x)

        train[test_index]=pre

        test_pre[i, :]= pred
        cv_scores_logloss.append(log_loss(te_y, pre))
        cv_scores.append(f1_score(te_y, np.argmax(pre,axis=1),labels=range(0,19),average='macro'))

        print("%s now score is:"%clf_name,cv_scores)
    test[:]=test_pre.mean(axis=0)
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
        f.write("%s_logloss_score_mean:"%clf_name+str(np.mean(cv_scores_logloss))+"\n")
    return train.reshape(-1,class_num),test.reshape(-1,class_num),np.mean(cv_scores)


def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lgb",19)
    return xgb_train, xgb_test,cv_scores

def lr(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lr",19)
    return xgb_train, xgb_test,cv_scores

import lightgbm
from sklearn.cross_validation import KFold
folds = 5
seed = 2018


kf = KFold(train_x.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test,m=lgb(train_x, train_y, test_x)

score=f1_score(train_y, np.argmax(lgb_train,axis=1),labels=range(0,19),average='macro')
score=str(score)[:7]
print(score)

loss_score=log_loss(train_y,lgb_train)
loss_score=str(loss_score)[:7]

train_pred=pd.DataFrame(lgb_train)
train_pred.columns=["class_%s"%i for i in range(19)]
train_pred["label"]=train_y
train_pred.to_csv("../sub/train_level2_prob_lgb.csv",index=None)

test_pred=pd.DataFrame(lgb_test)
test_pred.columns=["class_%s"%i for i in range(19)]
test_pred.to_csv("../sub/test_level2_prob_lgb.csv",index=None)
#生成提交结果
preds=np.argmax(lgb_test,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
test_pred["id"]=range(102277)
test_pred[["id","class"]].to_csv('../sub/sub_stacking_level2_%s_%s.csv'%(score,loss_score),index=None)
t2=time.time()
print("time use:",t2-t1)

