import pandas as pd
import numpy as np
import copy
import pickle


lgb = pd.read_pickle("feature_cv10/b1.pkl")
b1 = lgb.copy()
col = lgb.columns

col = col.tolist()

dic={
    "0":1,
    "1":1,
    "2":1,
    "3":1,
    "4":1,
    "5":1,
    "6":1,
    "7":1,
    "8":1,
    "9":1,
    "10":1,
    "11":1,
    "12":1,
    "13":1,
    "14":1,
    "15":1,
    "16":1,
    "17":1,
    "18":1,
}
count={
3:8313,
13:7907,
9: 7675,
15:7511,
18:7066,
8: 6972,
6: 6888,
14:6740,
19:5524,
1: 5375,
12:5326,
10:4963,
4: 3824,
11:3571,
16:3220,
17:3094,
7: 3038,
2: 2901,
5: 2369,
}
times=5
for tt in range(times):
    for i in col:
        k=dic[i]
        b1[i] = lgb[i]*k
    preds = np.argmax(b1.values, axis=1)
    test_pred = pd.DataFrame(preds)
    test_pred.columns = ["class"]
    test_pred["class"] = (test_pred["class"] + 1).astype(int)
    if tt==0:
        print(test_pred["class"].value_counts())

    dict_count=(dict(test_pred["class"].value_counts()))
    dic_ori=copy.deepcopy(dic)
    #for i in col:
        #dic[i]=((count[int(i)+1]/dict_count[int(i)+1])-dic_ori[i])/2+dic_ori[i]
    for i in col:
        dic[i]=((count[int(i)+1]/dict_count[int(i)+1])-1)/0.97+dic_ori[i]


print(test_pred["class"].value_counts())
print(dic)
#"""
test=pd.read_csv("../input/test_set.csv")
test_pred["id"] = list(test["id"])
test_pred[["id", "class"]].to_csv('a_blend_it_all_re_min1_%s.csv'%times, index=None)

with open("process.txt","a") as f:
    f.write(str(test_pred["class"].value_counts())+"\n"+str(dic)+"\n"+str(times)+"\n"+"min1 f1*************************************\n")
#"""
