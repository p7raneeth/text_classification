from io import StringIO
import pandas as pd


def read_validate_data(fname:str, extension):
    """
    Validate the input dataframe on columns, rowns and extension.
    """
    flag=0   # both auto and custom ml are activated
    if extension not in ['csv', 'xlsx']:
        return ValueError('Invalid file format.')
    file_name = StringIO(str(fname.file.read(), 'utf-8'))

    df = pd.read_csv(file_name, engine='python', nrows=10000)
    print('-----------------', df.shape)
    if df.shape[1] <= 1 or df.shape[0] <= 50:
        flag = 1 # invalid dataframe
    elif df.shape[0] <= 1000:
        flag = 2 # only custom ml is activated
    return flag
    
