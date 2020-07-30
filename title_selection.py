#!/usr/bin/env python
# coding: utf-8

import gensim
import MeCab
import sys
import numpy as np
import pandas as pd
import codecs
import math
from sklearn.svm import SVC

import vdata

def pre_process(train_file):
    with codecs.open(train_file, "r", "UTF-8", "ignore") as file:  #"Shift-JIS"
        df = pd.read_table(file, delimiter=",")
    
    df=df.dropna()
    train_data = df.loc[:,"title"].values
    target_data = df.loc[:,"target"].values
    return train_data, target_data

# def vectorize_data(input_data):
#     model_gensim = gensim.models.KeyedVectors.load("models/w2v_model.bin")
#     vect_data = []
#     m = MeCab.Tagger("-Owakati")
#     for sentense in input_data:
#         temp = m.parse(sentense)
#         temp_list = temp.split()
#         sum = 0
#         for word in temp_list:
#             try: 
#                 sum = sum + model_gensim[word]
#             except:
#                 # print(word)
#                 pass
#         sum = sum/len(temp_list)
#         vect_data.append(sum)

#     vect_data = np.array(vect_data)
#     return vect_data

def build_model(vector_data300,target_data):
    clf = SVC(gamma='auto',probability=True,random_state=43)
    model = clf.fit(vector_data300, target_data)
    return model

def predict(test_file,svc1):
    df_test = pd.read_csv(test_file)
    test_data = df_test.loc[:,"title"].values

    vect_test_data = vdata.vectorize_data(test_data)    
    pred = svc1.predict_proba(vect_test_data)

    pred_pd =  pd.Series(pred[0:len(pred),1])
    test_pd = pd.Series(test_data)
    test_pred = pd.concat([test_pd, pred_pd] ,axis=1)
    test_pred.columns = ["title","score"]

    test_pred = test_pred.sort_values("score",ascending=False)
    result = test_pred[test_pred["score"] >0.6]
    
    test_pred.to_csv("data/result_0427.csv",mode="w",encoding="utf_8_sig")
    
    return result
    
# def main():
#     train_file = "train_target_data03010426.csv"
#     test_file = "matome_rss_20200427.csv"

#     train_data,target_data = pre_process(train_file)
#     vect_train_data = vdata.vectorize_data(train_data)
    
#     svc1 = build_model(vect_train_data, target_data)
#     result = predict(test_file, svc1)
    
#     count =0 
#     for i in result.index:
#         print(result.loc[i,"score"].round(4),result.title[i])
#         count += 1
#         if count >=10 :
#             break


if __name__ == '__main__':
    train_file = "data/train_target_data03010426.csv"
    test_file = "data/matome_rss_20200427.csv"

    train_data,target_data = pre_process(train_file)
    vect_train_data = vdata.vectorize_data(train_data)
    
    svc1 = build_model(vect_train_data, target_data)
    result = predict(test_file, svc1)
    
    count =0 
    for i in result.index:
        print(result.loc[i,"score"].round(4),result.title[i])
        count += 1
        if count >=10 :
            break