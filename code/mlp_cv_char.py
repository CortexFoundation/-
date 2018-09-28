# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
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

t1=time.time()
###################################################################################################################################
train = pd.read_csv('../input/train_set.csv')
test = pd.read_csv('../input/test_set.csv')
train_id = train[["id"]].copy()
test_id = test[["id"]].copy()

column="article"
n = train.shape[0]
vec = TfidfVectorizer(min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

X_train=trn_term_doc.tocsr()
X_test=test_term_doc.tocsr()

y_true=(train["class"]-1).astype(int)
y = np_utils.to_categorical(y_true)
###################################################################################################################################
# 建立模型
def get_model():
    inp = Input(shape=(X_train.shape[1],))

    mlp=Dense(256,activation="relu")(inp)
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

    history = model.fit(X_train_tr, y_tr, batch_size=128, epochs=100, verbose=1, validation_data=(X_train_te,y_te),
                        callbacks=[early_stop, check_point])

    model.load_weights('cate_model.hdf5')
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
train_prob.columns=["class_prob_%s"%i for i in range(1,test_pred.shape[1]+1)]
train_prob.to_csv('../sub_prob/train_prob_mlp_cv_%s.csv'%score,index=None)
#保存预测概率文件
test_prob=pd.DataFrame(test_pred)
test_prob.columns=["class_prob_%s"%i for i in range(1,test_pred.shape[1]+1)]
test_prob.to_csv('../sub_prob/test_prob_mlp_cv_%s.csv'%score,index=None)
