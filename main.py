# import tensorflow
import os
import string
from distutils import extension
from io import StringIO
import contractions as ct
import eli5
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import read_validate_data
from cleaning import *
from preprocessing import *
import re
from modelling_metrics import *


app_nlp = FastAPI()
global_vars = {}

validation_dict = {'0': 'Auto and Custom',
                    '1': 'Invalid',
                   '2': 'Custom only'}

@app_nlp.get('/')
async def sanity_check():
    pickle.load(open('finalized_model.pkl', 'rb'))
    return {'message': 'Hello World!'}

@app_nlp.post('/data_validation')
def data_validation(fname: UploadFile=File('IMDB Dataset.csv')):
    global_vars['fname'] = fname
    extension = fname.filename.split('.')[-1]
    flag, df = read_validate_data(fname, extension)
    global_vars['data'] = df
    #print("validation finished")
    return {'message': f'data_validation is waas completed successfully with flag {validation_dict[str(flag)]}'}


@app_nlp.get('/preview')
def data_preview(num_records:int):
    # #print(global_vars['data'].shape, 'in preview api')
    # #print(type(global_vars['data']))
    # #print(global_vars['data'].head(num_records))
    columns =  global_vars['data'].columns
    features, global_vars['target'] = columns[:-1], columns[-1]
    #print('****************************************************************')
    #print(global_vars['target'])
    #print(global_vars['data']['sentiment'])
    #print('****************************************************************')
    percent_num_vals = round(global_vars['data'].isnull().sum()/ global_vars['data'].shape[0],5)*100
    return {f" features : {features} -- columns : {columns} -- percent_num_vals: {percent_num_vals} -- head : { global_vars['data'].head(num_records) }"}


@app_nlp.get('/data_understanding')
def data_understanding() -> dict:
    data_properties = {}
    # #print('-------------  data understanding --------------------')
    # #print(global_vars['fname']) 
    # #print(str(global_vars['fname'].file.read(), 'utf-8'))
    mem_on_disk = os.path.getsize('/Users/praneeth/Desktop/projects/nlp/text_classification/IMDB Dataset.csv')
    data_properties['file_size (mb)'] = mem_on_disk/10**6
    #print(dict(global_vars['data']['sentiment'].value_counts()))
    #print(global_vars)
    data_properties['category_count'] = dict(global_vars['data'][global_vars['target']].value_counts())
    # data_properties['null_count'] = null count logic to be added
    # calculate the average number of words per sentence
    # calculate the length of longest sentence
    # calculate the length of shortest sentence 
    # most common words in the whole corpus
    # calculate top unique words in the whole corpus
    #print('*************')
    #print(data_properties)
    return {f'{data_properties}'}


@app_nlp.get('/data_cleaning')
async def data_clean():
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: x.lower())
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: x.replace('\n', ''))
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: ct.fix(x))
    # whitespace removal
    # hashtag removal
    # global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: " ".join(x.split()))
    # eliminate URLs and links
    # stopwords removal
    global_vars['data']['cleaned_review'] = global_vars['data']['review'].apply(lambda x : ' '.join(stopwordremoval(x, stop_words)))
    # remove numbers and junk characters
    global_vars['data']['cleaned_review'] = global_vars['data']['cleaned_review'].apply(lambda x : re.sub(r'[^a-zA-Z ]+', '', x))
    # stemming

    # lemmatization
    global_vars['data']['cleaned_review'] = global_vars['data']['cleaned_review'].apply(lambda x: " ".join(x.split()))
    #print(global_vars['data']['cleaned_review'][0])

    return {'message': 'cleaning completed successfully'}


@app_nlp.get('/pre_processing')
async def pre_processing(select_vectorizer: int = 2, split_percent:float = 0.25):
    global_vars['data_vectorizer'] = select_vectorizer
    # seperate X and y
    training_data, training_labels = global_vars['data']['cleaned_review'], global_vars['data']['sentiment']
    trainX, testX, trainY, testY = train_test_split(training_data, training_labels, test_size=split_percent)
    # perform TFIDF Vectorization
    if select_vectorizer == 1:
        # write logic to create vectors using TFIDF technique
        pass
    elif select_vectorizer == 2:
        # perform Word2Vec Vectorization
        X_train_vect_avg, X_test_vect_avg = Word2Vectorizer(trainX, trainY, testX, testY)
        global_vars['X_train_vect_avg'], global_vars['X_test_vect_avg'] = X_train_vect_avg, X_test_vect_avg
        global_vars['trainY'], global_vars['testY'] = trainY, testY
        return {'message': 'Vectorization completed successfully'}
    elif select_vectorizer == 3:
        # perform other vectorization methods
        pass
    else:
        pass

# model_dict = {

#     1 : 'Random Forest Model',
#     2 : 'Logistic Regression'

# }
@app_nlp.get('/modelling')
async def ml_modelling(select_model:int, select_metric:str):
    global_vars['model_selection'] = select_model
    if select_model == 1:
        X_train_predict, X_test_predict, training_labels, testing_labels = LogisticRegressionClassifier( global_vars['X_train_vect_avg'], global_vars['trainY'], global_vars['X_test_vect_avg'], global_vars['testY'])
        training_score, testing_score = compute_metrics(X_train_predict, X_test_predict, training_labels, testing_labels, select_metric)    
    
    elif select_model == 2:
        RandomForestClassifier()
        pass
    elif select_model == 3:
        GradientBoostingClassifier()
        pass

@app_nlp.post('/inference')
async def ml_inference(filename:str, X_inference:list):
    #print('****************************************')
    #print(X_inference)
    #print(type(X_inference))
    #print(global_vars['data_vectorizer'])
    #print(type(global_vars['data_vectorizer']))
    #print('--------------------------------------')
    cleaned_data = inf_clean(X_inference)
    
    if global_vars['data_vectorizer'] == 1:
        pass
    elif global_vars['data_vectorizer'] == 2:
        #print('************* data_vectorizer 2 ****************************')
        X_inf_vect = inf_word2vec(X_inference)
        print(type(X_inf_vect)) #list
        print(len(X_inf_vect))
        print(X_inf_vect)
        loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
        print('------loaded model type ----------------', type(loaded_model))
        inf_predictions =  loaded_model.predict(X_inf_vect)
        #print(inf_predictions)
        return inf_predictions
        

