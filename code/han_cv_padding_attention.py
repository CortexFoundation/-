# -*- coding:utf-8 -*-
"""
https://github.com/synthesio/hierarchical-attention-networks/blob/master/model.py
"""
import numpy as np
import pandas as pd
import time
import gc
from collections import defaultdict
from sklearn.metrics import mean_squared_error, f1_score
from gensim.models import word2vec
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Embedding, BatchNormalization, Bidirectional, Conv1D, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Input, Lambda, TimeDistributed, Convolution1D
from keras.layers import LSTM, concatenate, Conv2D,SpatialDropout1D,Bidirectional,CuDNNGRU
from keras.layers import AveragePooling1D, Flatten, GlobalMaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

t1 = time.time()
###################################################################################################################################
model = word2vec.Word2Vec.load("daguan.w2v")

train = pd.read_csv('../input/train_set.csv')  # [:1000]
test = pd.read_csv('../input/test_set.csv')  # [:1000]
test_id = test[["id"]].copy()

MAX_SEQUENCE_LENGTH = 2500
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 100

MAX_SENTS = 25
SENT_LENGTH = 100

column = "word_seg"
import copy
def pad(x):
    x=x.replace("129533 889228 520477 658131 490774 422098 816903 608182 1226448","")
    pad=copy.copy(x)
    length=len(x.split(" "))
    n=MAX_SEQUENCE_LENGTH//length
    for i in range(n):
        x=x+" "+pad
    result=" ".join(x.split(" ")[:MAX_SEQUENCE_LENGTH])
    return result

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, )
tokenizer.fit_on_texts(list(train[column]) + list(test[column]))

train[column]=train[column].apply(pad)
test[column]=test[column].apply(pad)

sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))

X_train = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))
print(nb_words)
ss = 0
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in model.wv.vocab:
        ss += 1
        if i >= nb_words:
            break
        embedding_matrix[i] = model.wv[word]
    else:
        # print word
        pass
print(ss)
print(embedding_matrix.shape)
# np.save("embedding_matrix.npy",embedding_matrix)

y_true = (train["class"] - 1).astype(int)
y = np_utils.to_categorical(y_true)


###################################################################################################################################
# 建立模型
from model_utils import AttentivePoolingLayer,Attention
def get_model():

    sentence_inp = Input(shape=(SENT_LENGTH,))
    emb = Embedding(nb_words, EMBEDDING_DIM, input_length=SENT_LENGTH, weights=[embedding_matrix],
                    trainable=False)(sentence_inp)
    bigru = Bidirectional(CuDNNGRU(128,return_sequences=True))(emb)
    bigru = TimeDistributed(Dense(256))(bigru)
    bigru = Attention(100)(bigru)
    sentEncoder = Model(sentence_inp, bigru)

    review_input = Input(shape=(MAX_SENTS, SENT_LENGTH))
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    bigru_sent = Bidirectional(CuDNNGRU(128, return_sequences=True))(review_encoder)
    bigru_sent = TimeDistributed(Dense(256))(bigru_sent)
    bigru_sent = Attention(25)(bigru_sent)

    fc1 = Dense(256, activation='relu')(bigru_sent)
    fc2 = Dense(128, activation='relu')(fc1)
    fc2 = BatchNormalization()(fc2)
    output = Dense(19, activation="softmax")(fc2)
    model = Model(inputs=review_input, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


from sklearn.cross_validation import KFold

folds = 5
seed = 2018
skf = KFold(train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

te_pred = np.zeros((X_train.shape[0], 19))
test_pred = np.zeros((X_test.shape[0], 19))
test_pred_cv = np.zeros((5, X_test.shape[0], 19))
cnt = 0
score = 0
score_cv_list = []
X_test=X_test.reshape(X_test.shape[0], MAX_SENTS, SENT_LENGTH)
for ii, (idx_train, idx_val) in enumerate(skf):
    X_train_tr = X_train[idx_train]
    X_train_te = X_train[idx_val]
    y_tr = y[idx_train]
    y_te = y[idx_val]

    X_train_tr=X_train_tr.reshape(X_train_tr.shape[0], MAX_SENTS, SENT_LENGTH)
    X_train_te=X_train_te.reshape(X_train_te.shape[0], MAX_SENTS, SENT_LENGTH)

    model = get_model()
    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint('cate_model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)

    history = model.fit(X_train_tr, y_tr, batch_size=128, epochs=10, verbose=1, validation_data=(X_train_te, y_te),
                        callbacks=[early_stop, check_point]
    )

    model.load_weights('cate_model.hdf5')
    preds_te = model.predict(X_train_te)
    score_cv = f1_score(y_true[idx_val], np.argmax(preds_te, axis=1), labels=range(0, 19), average='macro')
    score_cv_list.append(score_cv)
    print(score_cv_list)
    te_pred[idx_val] = preds_te
    preds = model.predict(X_test)
    test_pred_cv[ii, :] = preds
    # break
    del model
    gc.collect()
with open("score_cv.txt", "a") as f:
    f.write("%s now score is:" % "han" + str(score_cv_list) + "\n")

test_pred[:] = test_pred_cv.mean(axis=0)
print(te_pred)
score = f1_score(y_true, np.argmax(te_pred, axis=1), labels=range(0, 19), average='macro')
score = str(score)[:7]
print(score)
# 保存预测概率文件
train_prob = pd.DataFrame(te_pred)
train_prob.columns = ["class_prob_%s" % i for i in range(1, test_pred.shape[1] + 1)]
train_prob.to_csv('../sub_prob/train_prob_han_cv_%s.csv' % score, index=None)
# 保存预测概率文件
test_prob = pd.DataFrame(test_pred)
test_prob.columns = ["class_prob_%s" % i for i in range(1, test_pred.shape[1] + 1)]
test_prob.to_csv('../sub_prob/test_prob_han_cv_%s.csv' % score, index=None)
#生成提交结果
preds=np.argmax(test_pred,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('../sub/han_cv_%s.csv'%score,index=None)

