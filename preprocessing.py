from xmlrpc.client import boolean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim
import numpy as np
import pandas as pd

def TFIDFVectorizer(trainX, trainY, testX, testY , inf=False):
    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
    X_train_text = text_transformer.fit_transform(trainX)
    X_test_text = text_transformer.transform(testX)
    return(X_train_text, X_test_text)

def Word2Vectorizer(trainX, trainY, testX, testY, save_model:boolean = True):

    training_data = trainX.apply(gensim.utils.simple_preprocess)
    testing_data = testX.apply(gensim.utils.simple_preprocess)
    model = Word2Vec(training_data, window=10, min_count=2, workers=4)
    if save_model:
        model.save('word2vec_trainedmodel.model')
    words = set(model.wv.index_to_key)
    X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in training_data])
    X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in testing_data])
    X_train_vect_avg, X_test_vect_avg = avg_w2v(X_train_vect, X_test_vect)
    return(X_train_vect_avg, X_test_vect_avg)
     
def inf_word2vec(cleaned_data):
    #print('start word2vec')
    #print(cleaned_data)
    cleaned_data_vectors = pd.Series(cleaned_data).apply(gensim.utils.simple_preprocess)
    model = Word2Vec.load('word2vec_trainedmodel.model')
    words = set(model.wv.index_to_key)
    inf_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in cleaned_data_vectors])
    inf_vect_avg = inf_avg_w2v(inf_vect)
    #print('ending word2vec')
    #print(inf_vect_avg)
    return(inf_vect_avg)

def avg_w2v(X_train_vect, X_test_vect):
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    return(X_train_vect_avg, X_test_vect_avg)
    

def inf_avg_w2v(inf_vect):
    inf_vect_avg = []
    for v in inf_vect:
        if v.size:
            inf_vect_avg.append(v.mean(axis=0))
        else:
            inf_vect_avg.append(np.zeros(100, dtype=float))
    return(inf_vect_avg)
