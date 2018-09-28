import os
import copy
import pandas as pd
from gensim.models import Word2Vec
from random import shuffle
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import logging
import time
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO)

train = pd.read_csv('../input/train_set.csv')
test = pd.read_csv('../input/test_set.csv')

all_samples = pd.concat([
    train,
    test,
]).reset_index(drop=True)
print(all_samples.shape)
"""
all_samples = all_samples['word_seg'].values

all_samples = [text_to_word_sequence(text) for text in tqdm(all_samples)]
gc.collect()
"""
seq=all_samples['word_seg'].str.split().tolist()

model = Word2Vec(seq,size=300, window=5,workers=16,min_count=1,iter=40)
model.save('daguan_40.w2v')

