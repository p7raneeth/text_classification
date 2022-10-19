# import tensorflow
from distutils import extension
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.feature_extraction import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import eli5
from fastapi import *

from data import read_validate_data


app_nlp = FastAPI()



@app_nlp.get('/')
async def sanity_check():
    return {'message': 'Hello World!'}


@app_nlp.post('/data_validation')

def data_validation(fname: UploadFile=File(None)):
    extension = fname.filename.split('.')[-1]
    flag = read_validate_data(fname, extension)
    print("validation finished")
    return {'message': f'data_validation is completed successfully with flag {validation_dict[str(flag)]}'}

validation_dict = {'0': 'Auto and Custom',
                    '1': 'Invalid',
                   '2': 'Custom only'}
