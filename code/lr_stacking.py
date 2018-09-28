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
from sklearn.preprocessing import OneHotEncoder

import time
#lda特征
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
path="../"
pickle_path="pickle/"
def get_lda_feature(sample,column,value):
    sample_filename = "_".join(["lda_features",column,str(value), str(len(sample))]) + ".pkl"
    try:
        with open(path + pickle_path + sample_filename, "rb") as fp:
            print("load lda feature from pickle_all_data file:on: {}...".format(column+"_"+str(value)))
            col = pickle.load(fp)
        for c in col.columns:
            sample[c] = col[c]
    except:
        print("get lda feature from pickle_all_data file:on: {}...".format(column+"_"+str(value)))

        cv = CountVectorizer(max_features=1000)
        tf = cv.fit_transform(sample[column])
        lda = LatentDirichletAllocation(n_components=value,learning_method='batch',n_jobs=-1,random_state=2018)
        lda_features=lda.fit_transform(tf)
        new_columns=["%s_lda_%s"%(column,i) for i in range(value)]
        lda_features = pd.DataFrame(lda_features, columns=new_columns).reset_index(drop=True)
        for c in lda_features.columns:
            sample[c]=lda_features[c]
        with open(path + pickle_path + sample_filename, "wb") as fp:
            col = sample[new_columns]
            pickle.dump(col, fp)
    return sample


t1=time.time()
train = pd.read_csv('../input/train_set.csv')#[:10000]
test = pd.read_csv('../input/test_set.csv')#[:10000]
print(train.shape)
print(test.shape)
train_id = train[["id"]].copy()
test_id = test[["id"]].copy()
train_cnt=train.shape[0]

column="word_seg"
value=38
sample=train[["word_seg","article"]].copy().append(test[["word_seg","article"]].copy())
sample=get_lda_feature(sample,column,value)

#sample["char_len"]=sample["article"].apply(lambda x:len(x.split(" ")))
#sample["char_most"]=sample["article"].apply(lambda x:Counter(x.split(" ")).most_common(1)[0][1])
#sample["word_len"]=sample["word_seg"].apply(lambda x:len(x.split(" ")))
#sample["word_most"]=sample["word_seg"].apply(lambda x:Counter(x.split(" ")).most_common(1)[0][1])
#sample["word_rt"]=sample["word_len"]/sample["word_most"]
#sample["char_rt"]=sample["char_len"]/sample["word_most"]
"""
for i in ["520477","1033823","816903","995362","701424","834740"]:
    sample["%s_count"%i] = sample["word_seg"].apply(lambda x: x.split(" ").count(i))
for i in ["1033823","816903","995362","701424","834740"]:
    sample["%s_count_rt"%i] = sample["%s_count"%i]/sample["520477_count"]
"""


del sample["word_seg"]
del sample["article"]
gc.collect()

train_x=sample[:train_cnt].values
test_x=sample[train_cnt:].values

#stacking


file_list=[]
train_textcnn=[]
test_textcnn=[]
train_rnn=[]
test_rnn=[]
train_att=[]
test_att=[]
train_mlp=[]
test_mlp=[]
train_fasttext=[]
test_fasttext=[]
train_han=[]
test_han=[]
train_avgbigru=[]
test_avgbigru=[]
train_cap=[]
test_cap=[]
train_lstm=[]
test_lstm=[]
train_bigruatt=[]
test_bigruatt=[]
for i in os.listdir("../sub_prob/"):
    if ("train" in i) and ("#" not in i):
        file_list.append("../sub_prob/"+i)
for file in file_list:
    if ("roc_select_cv" in file) or ("knn100_select_cv" in file):
        train_stacking=pd.read_csv(file)[["class_prob_1"]].values
        test_stacking=pd.read_csv(file.replace("train","test"))[["class_prob_1"]].values
        enc = OneHotEncoder()
        enc.fit(train_stacking)
        train_stacking=enc.transform(train_stacking).toarray()
        test_stacking=enc.transform(test_stacking).toarray()
    elif "ovr" in file:
        train_stacking=pd.read_csv(file).values
        test_stacking=pd.read_csv(file.replace("train","test")).values
    else:
        train_stacking=pd.read_csv(file)[["class_prob_%s"%i for i in range(1,20)]].values
        test_stacking=pd.read_csv(file.replace("train","test"))[["class_prob_%s"%i for i in range(1,20)]].values
        if "_textcnn_cv_" in file:
            train_textcnn.append(train_stacking)
            test_textcnn.append(test_stacking)
            continue
        if "_rnn_cv_" in file:
            train_rnn.append(train_stacking)
            test_rnn.append(test_stacking)
            continue
        if "_att_cv_" in file:
            train_att.append(train_stacking)
            test_att.append(test_stacking)
            continue
        if "_mlp_cv_" in file:
            train_mlp.append(train_stacking)
            test_mlp.append(test_stacking)
            continue
        if "_fasttext_cv_" in file:
            train_fasttext.append(train_stacking)
            test_fasttext.append(test_stacking)
            continue
        if "_han_cv_" in file:
            train_han.append(train_stacking)
            test_han.append(test_stacking)
            continue
        if "_avgbigru_cv_" in file:
            train_avgbigru.append(train_stacking)
            test_avgbigru.append(test_stacking)
            continue
        if "_cap_cv_" in file:
            train_cap.append(train_stacking)
            test_cap.append(test_stacking)
            continue
        if "_lstm_cv_" in file:
            train_lstm.append(train_stacking)
            test_lstm.append(test_stacking)
            continue
        if "_bigruatt_cv_" in file:
            train_bigruatt.append(train_stacking)
            test_bigruatt.append(test_stacking)
            continue
    #train_pred=np.argmax(train_stacking,axis=1).reshape((-1,1))
    #test_pred=np.argmax(test_stacking,axis=1).reshape((-1,1))
    train_x=np.concatenate([train_x,train_stacking],axis=1)
    test_x=np.concatenate([test_x,test_stacking],axis=1)
train_textcnn=np.mean(train_textcnn,axis=0)
test_textcnn=np.mean(test_textcnn,axis=0)
train_rnn=np.mean(train_rnn,axis=0)
test_rnn=np.mean(test_rnn,axis=0)
train_att=np.mean(train_att,axis=0)
test_att=np.mean(test_att,axis=0)
train_mlp=np.mean(train_mlp,axis=0)
test_mlp=np.mean(test_mlp,axis=0)
train_fasttext=np.mean(train_fasttext,axis=0)
test_fasttext=np.mean(test_fasttext,axis=0)
train_han=np.mean(train_han,axis=0)
test_han=np.mean(test_han,axis=0)
train_avgbigru=np.mean(train_avgbigru,axis=0)
test_avgbigru=np.mean(test_avgbigru,axis=0)
train_cap=np.mean(train_cap,axis=0)
test_cap=np.mean(test_cap,axis=0)
train_lstm=np.mean(train_lstm,axis=0)
test_lstm=np.mean(test_lstm,axis=0)
train_bigruatt=np.mean(train_bigruatt,axis=0)
test_bigruatt=np.mean(test_bigruatt,axis=0)
train_x = np.concatenate([train_x, train_textcnn,train_rnn,train_att,train_mlp,train_fasttext,train_avgbigru,train_cap,train_lstm,train_bigruatt], axis=1)
test_x = np.concatenate([test_x, test_textcnn,test_rnn,test_att,test_mlp,test_fasttext,test_avgbigru,test_cap,test_lstm,test_bigruatt], axis=1)
#"""
print(train_x.shape)
print(test_x.shape)

train_y=(train["class"]-1).astype(int)
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
            model = LogisticRegression(C=4, dual=False,random_state=2018)
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
lgb_train, lgb_test,m=lr(train_x, train_y, test_x)

score=f1_score(train_y, np.argmax(lgb_train,axis=1),labels=range(0,19),average='macro')
score=str(score)[:7]
print(score)

loss_score=log_loss(train_y,lgb_train)
loss_score=str(loss_score)[:7]

train_pred=pd.DataFrame(lgb_train)
train_pred.columns=["class_%s"%i for i in range(19)]
train_pred["label"]=train_y
train_pred.to_csv("../sub/train_prob_lr_v2.csv",index=None)

test_pred=pd.DataFrame(lgb_test)
test_pred.columns=["class_%s"%i for i in range(19)]
test_pred.to_csv("../sub/test_prob_lr_v2.csv",index=None)
#生成提交结果
preds=np.argmax(lgb_test,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('../sub/sub_stacking_lr_v2_%s_%s.csv'%(score,loss_score),index=None)
t2=time.time()
print("time use:",t2-t1)

