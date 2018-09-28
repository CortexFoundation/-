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

train = pd.read_csv('../input/train_set.csv')#[:10000]
train_svm_2000=pd.read_csv("../sub_prob/train_prob_svm_cv_2000_0.77359.csv")
del train_svm_2000["id"]
ss=train_svm_2000.values
train["class"]=(train["class"]-1).astype(int)
train["pre"]=np.argmax(ss,axis=1)

train[["class","pre"]].to_csv("ana.csv",index=None)
