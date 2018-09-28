# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
import pickle
import gc
import os
from collections import defaultdict
from sklearn.metrics import mean_squared_error,f1_score
from gensim.models import word2vec
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Activation, Dropout, Embedding,BatchNormalization,Bidirectional,Conv1D,GlobalMaxPooling1D,GlobalAveragePooling1D,Input,Lambda,TimeDistributed,Convolution1D
from keras.layers import LSTM,concatenate
from keras.layers import AveragePooling1D,Flatten
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

###################################################################################################################################
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

X_train=sample[:train_cnt].values
X_test=sample[train_cnt:].values

#stacking


#"""
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
for i in os.listdir("../sub_prob/"):
    if ("train" in i) and ("#" not in i):
        file_list.append("../sub_prob/"+i)
for file in file_list:
    if "ovr" in file:
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
    #train_pred=np.argmax(train_stacking,axis=1).reshape((-1,1))
    #test_pred=np.argmax(test_stacking,axis=1).reshape((-1,1))
    X_train=np.concatenate([X_train,train_stacking],axis=1)
    X_test=np.concatenate([X_test,test_stacking],axis=1)
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
X_train = np.concatenate([X_train, train_textcnn,train_rnn,train_att,train_mlp,train_fasttext], axis=1)
X_test = np.concatenate([X_test, test_textcnn,test_rnn,test_att,test_mlp,test_fasttext], axis=1)
#"""
print(X_train.shape)
print(X_test.shape)

y_true=(train["class"]-1).astype(int)
y = np_utils.to_categorical(y_true)
###################################################################################################################################
# 建立模型
def get_model():
    inp = Input(shape=(X_train.shape[1],))
    mlp=Dropout(0.2)(inp)
    mlp=Dense(256,activation="relu")(mlp)
    mlp=Dropout(0.2)(mlp)

    output=Dense(19,activation="softmax")(mlp)
    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer=Adam(lr=0.002), loss="categorical_crossentropy")
    model.summary()
    return model

from sklearn.cross_validation import KFold
folds = 5
seed = 2018
skf = KFold(train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

te_pred=np.zeros((X_train.shape[0],19))
test_pred=np.zeros((X_test.shape[0],19))
test_pred_cv=np.zeros((5,X_test.shape[0],19))
cnt=0
score=0
score_cv_list=[]
for ii,(idx_train, idx_val) in enumerate(skf):
    X_train_tr=X_train[idx_train]
    X_train_te=X_train[idx_val]
    y_tr=y[idx_train]
    y_te=y[idx_val]

    model = get_model()
    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint('cate_model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)

    history = model.fit(X_train_tr, y_tr, batch_size=1024, epochs=100, verbose=1, validation_data=(X_train_te,y_te),
                        #callbacks=[early_stop, check_point]
    )

    #model.load_weights('cate_model.hdf5')
    preds_te = model.predict(X_train_te)
    score_cv = f1_score(y_true[idx_val], np.argmax(preds_te, axis=1), labels=range(0, 19), average='macro')
    score_cv_list.append(score_cv)
    print(score_cv_list)
    te_pred[idx_val] = preds_te
    preds = model.predict(X_test)
    test_pred_cv[ii, :] = preds
    #break
with open("score_cv.txt", "a") as f:
    f.write("%s now score is:" % "mlp" + str(score_cv_list) + "\n")

test_pred[:]=test_pred_cv.mean(axis=0)
print(te_pred)
score=f1_score(y_true, np.argmax(te_pred,axis=1),labels=range(0,19),average='macro')
score=str(score)[:7]
print(score)
#保存预测概率文件
train_prob=pd.DataFrame(te_pred)
train_prob.columns=["class_%s"%i for i in range(19)]
train_prob.to_csv('../sub/train_prob_nn.csv',index=None)
#保存预测概率文件
test_prob=pd.DataFrame(test_pred)
test_prob.columns=["class_%s"%i for i in range(19)]
test_prob.to_csv('../sub/test_prob_nn.csv',index=None)
