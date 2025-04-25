# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 20:38:16 2025

@author: foresight
"""

import pandas as pd



df = pd.read_csv("../外部資料集/38_Private_Test_Set_and_Submission_Template_V2/private_x.csv")
df.to_parquet('../外部資料集/private.parquet', engine='pyarrow') 


df = pd.read_csv("../外部資料集/38_Public_Test_Set_and_Submmision_Template_V2/public_x.csv")
df.to_parquet('../外部資料集/public.parquet', engine='pyarrow')  


df = pd.read_csv("../外部資料集/38_Training_Data_Set_V2/training.csv")
df.to_parquet('../外部資料集/train.parquet', engine='pyarrow') 
