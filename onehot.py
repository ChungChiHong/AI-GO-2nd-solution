# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:01:41 2025

@author: foresight
"""

import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

    
def one_hot(df):
    
    df_ = pd.DataFrame()
    
    c = df.columns
    max_id = df[c].max(skipna=True).max()
    bar = tqdm(enumerate(range(int(max_id)+1)),total=int(max_id)+1)
    for idx,i in bar:

        df_[f"券商代號{i}"] = (df[c] == i).sum(axis=1)
        
    return df_



    return df
if __name__ == "__main__":
    
    columns = pd.read_csv('../外部資料集/columns.csv')
    cols = columns.columns
    c = [ i for i in cols if "券商代號" in i]


    df = pd.read_parquet("../外部資料集/train.parquet", columns=c, engine='pyarrow') 
    df_ = one_hot(df)
    df_.to_parquet('../外部資料集/train_onehot.parquet', engine='pyarrow') 

    df = pd.read_parquet("../外部資料集/public.parquet", columns=c, engine='pyarrow') 
    df_ = one_hot(df)
    df_.to_parquet('../外部資料集/public_onehot.parquet', engine='pyarrow') 

    df = pd.read_parquet("../外部資料集/private.parquet", columns=c, engine='pyarrow') 
    df_ = one_hot(df)
    df_.to_parquet('../外部資料集/private_onehot.parquet', engine='pyarrow') 
   









