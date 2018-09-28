# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from gensim.models import word2vec
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Dropout, Embedding,BatchNormalization,Bidirectional,Conv1D,GlobalMaxPooling1D,AveragePooling1D,AveragePooling2D,Convolution1D,Convolution2D
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import np_utils
t1=time.time()

model = word2vec.Word2Vec.load("daguan.w2v")

train = pd.read_csv('../input/train_set.csv')#[:1000]
test = pd.read_csv('../input/test_set.csv')#[:1000]
test_id=test[["id"]].copy()

MAX_SEQUENCE_LENGTH = 5000
MAX_NB_WORDS = 500000
EMBEDDING_DIM = 100

column="word_seg"
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS,)
tokenizer.fit_on_texts(list(train[column])+list(test[column]))

sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train = pad_sequences(sequences_all, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1
print(nb_words)
ss=0
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in model.wv.vocab:
        ss+=1
        embedding_matrix[i] = model.wv[word]
    else:
        #print word
        pass
print(ss)
print(embedding_matrix)

y=(train["class"]-1).astype(int)
y = np_utils.to_categorical(y)

# 建立模型
#'''
model = Sequential()
model.add(Embedding(nb_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,weights=[embedding_matrix],trainable=False))
model.add(AveragePooling2D((10,1),5))
model.add(AveragePooling2D((10,1),5))
model.add(Convolution1D(256,3,border_mode="same",activation="relu"))

#model.add(BatchNormalization())
model.add(Dense(256,activation="relu"))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())
model.add(Dense(128,activation="relu"))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())
model.add(Dense(64,activation="relu"))
#model.add(BatchNormalization())
model.add(Dense(19,activation="softmax"))
model.compile(optimizer=Adam(lr=0.002), loss="categorical_crossentropy")
model.summary()

early_stop = EarlyStopping(patience=2)
check_point = ModelCheckpoint('cate_model.hdf5', monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)

history = model.fit(X_train, y, batch_size = 1024, epochs = 30,verbose = 1, validation_split=0.1,callbacks=[early_stop,check_point])

model.load_weights('cate_model.hdf5')
preds = model.predict(X_test,verbose=1)

#保存概率文件
test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('../sub_prob/prob_lstm_baseline.csv',index=None)

#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('../sub/sub_cnn_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)


