#encoding=utf8
import pandas as pd
import numpy as np
import time
import gc
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time

t1=time.time()


print('loading train...')
df_train = pd.read_csv('../input/train_set.csv')#[:10000]
df_train["classify"]=df_train["classify"]-1
print('loading test')
df_test = pd.read_csv('../input/test_set.csv')#[:10000]
df=df_train.append(df_test).reset_index(drop=True)
train_length=df_train.shape[0]

from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences

max_features = 100000 # max amount of words considered
max_len = 500 #maximum length of text
dim = 100 #dimension of embedding
col = 'word_seg'

print('tokenizing...',end='')
tic = time.time()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df[col].values))
toc = time.time()
print('done. {}'.format(toc-tic))


print("   Transforming {} to seq...".format(col))
tic = time.time()
df[col] = tokenizer.texts_to_sequences(df[col])
toc = time.time()
print('done. {}'.format(toc-tic))

print('padding X_train')
tic = time.time()
X_train = pad_sequences(df[:train_length][col], maxlen=max_len,padding='post')
toc = time.time()
print('done. {}'.format(toc-tic))
#train_id=df[:train_length][["item_id"]].copy()

print('padding X_test')
tic = time.time()
X_test = pad_sequences(df[train_length:][col], maxlen=max_len,padding='post')
toc = time.time()
print('done. {}'.format(toc-tic))
test_id=df[train_length:][["id"]].copy()

cate_num=int(df["classify"].max()+1)
y = df[:train_length]["classify"].values
y = np_utils.to_categorical(y)
gc.collect()

import numpy as np
from keras.layers import Input,PReLU,BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding,Dropout
from keras.layers import Concatenate, Flatten, Bidirectional
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy



def all_pool(tensor):
    avg_tensor = GlobalAveragePooling1D()(tensor)
    max_tensor = GlobalMaxPooling1D()(tensor)
    res_tensor = Concatenate()([avg_tensor, max_tensor])
    return res_tensor

def build_model():
    inp = Input(shape=(max_len,))

    embedding = Embedding(max_features + 1, dim)(inp)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(embedding)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(x)
    x = all_pool(x)
    #x = BatchNormalization()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(cate_num, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(lr=0.002), loss="categorical_crossentropy")
    return model

model = build_model()
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
test_prob.to_csv('../sub_prob/prob_lr_baseline.csv',index=None)

#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('../sub/sub_lr_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)
