import gensim
import MeCab
import numpy as np
import pandas as pd


def vectorize_data(input_data):
    model_gensim = gensim.models.KeyedVectors.load("models/w2v_model.bin")
    vect_data = []
    m = MeCab.Tagger("-Owakati")
    for sentense in input_data:
        temp = m.parse(sentense)
        temp_list = temp.split()
        sum = 0
        for word in temp_list:
            try: 
                sum = sum + model_gensim[word]
            except:
                # print(word)
                pass
        sum = sum/len(temp_list)
        vect_data.append(sum)

    vect_data = np.array(vect_data)
    return vect_data
