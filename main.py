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
from sklearn.feature_extraction import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import read_validate_data
from cleaning import *

app_nlp = FastAPI()
global_vars = {}

validation_dict = {'0': 'Auto and Custom',
                    '1': 'Invalid',
                   '2': 'Custom only'}


@app_nlp.get('/')
async def sanity_check():
    return {'message': 'Hello World!'}


@app_nlp.post('/data_validation')
def data_validation(fname: UploadFile=File(None)):
    global_vars['fname'] = fname
    extension = fname.filename.split('.')[-1]
    flag, df = read_validate_data(fname, extension)
    global_vars['data'] = df
    print("validation finished")
    return {'message': f'data_validation is waas completed successfully with flag {validation_dict[str(flag)]}'}


@app_nlp.get('/preview')
def data_preview(num_records:int):
    # print(global_vars['data'].shape, 'in preview api')
    # print(type(global_vars['data']))
    # print(global_vars['data'].head(num_records))
    columns =  global_vars['data'].columns
    features, global_vars['target'] = columns[:-1], columns[-1]
    percent_num_vals = round(global_vars['data'].isnull().sum()/ global_vars['data'].shape[0],5)*100
    return {f" features : {features} -- columns : {columns} -- percent_num_vals: {percent_num_vals} -- head : { global_vars['data'].head(num_records) }"}


@app_nlp.get('/data_understanding')
def data_understanding() -> dict:
    data_properties = {}
    # print('-------------  data understanding --------------------')
    # print(global_vars['fname']) 
    # print(str(global_vars['fname'].file.read(), 'utf-8'))
    mem_on_disk = os.path.getsize('/Users/praneeth/Desktop/projects/nlp/text_classification/IMDB Dataset.csv')
    data_properties['file_size (mb)'] = mem_on_disk/10**6
    print(dict(global_vars['data']['sentiment'].value_counts()))
    print(global_vars)
    data_properties['category_count'] = dict(global_vars['data'][global_vars['target']].value_counts())
    # data_properties['null_count'] = null count logic to be added
    # average number of words per sentence
    # longest sentence length
    # shortest sentence length
    # most common words in the whole corpus
    print('*************')
    print(data_properties)
    return {f'{data_properties}'}


@app_nlp.get('/data_cleaning')
async def data_clean():
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: x.lower())
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: x.replace('\n', ''))
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: ct.fix(x))
    # whitespace removal
    global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: " ".join(x.split()))
    # hashtag removal
    # global_vars['data']['review'] = global_vars['data']['review'].apply(lambda x: " ".join(x.split()))
    # eliminate URLs and links
    # stopwords removal
    global_vars['data']['cleaned_review'] = global_vars['data']['review'].apply(lambda x : ' '.join(stopwordremoval(x, stop_words)))
    # remove numbers 
    # remove junk characters
    # stemming
    # lemmatization
    print(global_vars['data']['cleaned_review'][0])

    return {'message': 'cleaning completed successfully'}


@app_nlp.get('/data_vectorization')
async def data_classification():
    pass
