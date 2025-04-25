
import os
from joblib import dump#, load 
import numpy as np
from sklearn.model_selection import StratifiedKFold#, StratifiedGroupKFold
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline
import xgboost as xgb
import pandas as pd
import time
import gc
from sklearn.metrics import f1_score, precision_score, recall_score
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
import random
import warnings
warnings.filterwarnings("ignore")


# In[ ]:
    
class CFG:
  
    predict = 1

    # feature
    select_col = ['上市加權指數收盤價', '上市加權指數前1天收盤價', '上市加權指數前2天收盤價', '上市加權指數前3天收盤價', '上市加權指數前4天收盤價', '上市加權指數前5天收盤價', '上市加權指數前6天收盤價', '上市加權指數前7天收盤價', '上市加權指數前8天收盤價', '上市加權指數前9天收盤價', '上市加權指數前10天收盤價', '上市加權指數前11天收盤價', '上市加權指數前12天收盤價', '上市加權指數前13天收盤價', '上市加權指數前14天收盤價', '上市加權指數前15天收盤價', '上市加權指數前16天收盤價', '上市加權指數前17天收盤價', '上市加權指數前18天收盤價', '上市加權指數前19天收盤價', '上市加權指數前20天收盤價', '上市加權指數1天報酬率', '上市加權指數5天報酬率', '上市加權指數10天報酬率', '上市加權指數20天報酬率', '上市加權指數5天波動度', '上市加權指數10天波動度', '上市加權指數20天波動度', '上市加權指數成交量', '上市加權指數前1天成交量', '上市加權指數前2天成交量', '上市加權指數前3天成交量', '上市加權指數前4天成交量', '上市加權指數前5天成交量', '上市加權指數前6天成交量', '上市加權指數前7天成交量', '上市加權指數前8天成交量', '上市加權指數前9天成交量', '上市加權指數前10天成交量', '上市加權指數前11天成交量', '上市加權指數前12天成交量', '上市加權指數前13天成交量', '上市加權指數前14天成交量', '上市加權指數前15天成交量', '上市加權指數前16天成交量', '上市加權指數前17天成交量', '上市加權指數前18天成交量', '上市加權指數前19天成交量', '上市加權指數前20天成交量', '上市加權指數5天成交量波動度', '上市加權指數10天成交量波動度', '上市加權指數20天成交量波動度']

    # model cfg
    xgb_params = {
                'objective':'binary:logistic', 
                'enable_categorical':True,
                'learning_rate':0.1, 
                'n_estimators':200,
                'max_depth': 8,
                }
    
    # kfold
    seed = 69
    kfold = 5
  
    
# In[ ]:
    
def init_logger(log_file):
    
    with open(log_file, 'w') as f:
        for attr in dir(CFG):
            if not callable(getattr(CFG, attr)) and not attr.startswith("__"):
                f.write('{} = {}\n'.format(attr, getattr(CFG, attr)))
                
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=42, cudnn_deterministic=True):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    

# In[ ]:
    
if __name__ == "__main__":
    
    for dir_ in ["submission","log","model","predict"]:
        os.makedirs(dir_, exist_ok=True)
        
    exp_list = os.listdir("./model/")
    exp_id = 0
    for exp_name in exp_list:
        if int(exp_name.split("_")[0].split(".")[0])>exp_id:
            exp_id=int(exp_name.split("_")[0].split(".")[0])
        
    exp_id += 1
    print(f"exp id:{exp_id}")
    set_seed(CFG.seed)
    Logger = init_logger(f"./log/{exp_id}.txt")
    
    # data processing
    if CFG.select_col:
        df = pd.read_parquet("../外部資料集/train.parquet", columns=CFG.select_col+["飆股","ID"], engine='pyarrow') 
    else:
        df = pd.read_parquet("../外部資料集/train.parquet", engine='pyarrow') 
    
    category_cols = [c for c in df.columns if "代號" in c]
    df[category_cols] = df[category_cols].astype('category')
    
    if CFG.predict:
        X_test = pd.read_parquet('../外部資料集/public.parquet', engine='pyarrow')
        X_test_private = pd.read_parquet('../外部資料集/private.parquet', engine='pyarrow') 
        X_test = pd.concat([X_test, X_test_private], axis=0)
        X_test = X_test[CFG.select_col]
        category_cols = [c for c in X_test.columns if "代號" in c]
        X_test[category_cols] = X_test[category_cols].astype('category')
      
    X_train = df.drop(columns=["飆股"])
    y_train = df[["飆股","ID"]]
    del df
    gc.collect()        
    
    
    # model setting
    model = [] 
    pipeline = []
    pipeline.append(('classifier', xgb.XGBClassifier(**CFG.xgb_params)))
    xgb_model = Pipeline(pipeline)
    model.append(('xgb', xgb_model)) 
    estimator = VotingClassifier(model, voting='soft')
        
    
    # training
    print("Start training ...")
    start = time.time()
    
    estimators = []
    preds = []
    y_vailds = []
    
    skf = StratifiedKFold(n_splits=CFG.kfold,random_state=CFG.seed, shuffle=True)
    for fold, ( _, val_) in enumerate(skf.split(X=X_train, y=y_train["飆股"])):
          X_train.loc[val_ , "fold"] = fold
          y_train.loc[val_ , "fold"] = fold
          
    for fold in range(CFG.kfold):
    
        X_train_fold = X_train[X_train.fold!=fold].reset_index(drop=True).drop(columns=["fold","ID"])
        y_train_fold = y_train[y_train.fold!=fold].reset_index(drop=True).drop(columns=["fold","ID"])
        
        X_valid_fold = X_train[X_train.fold==fold].reset_index(drop=True).drop(columns=["fold","ID"])
        y_valid_fold = y_train[y_train.fold==fold].reset_index(drop=True).drop(columns=["fold","ID"])
    
        estimator.fit(X_train_fold, y_train_fold)
        dump(estimator, f'model/{exp_id}_{fold}.joblib')  
        y_pred_fold = estimator.predict_proba(X_valid_fold)[:, 1]
    
        preds.append(y_pred_fold)
        estimators.append(estimator)
        y_vailds.append(y_valid_fold)
        
        valid = X_train[X_train.fold==fold].reset_index(drop=True)[["ID"]]
        valid["飆股"] = y_valid_fold
        valid["p"] = y_pred_fold
        valid.to_csv(f'./predict/{exp_id}_{fold}.csv',index=False)
        
       
        Logger.info("-"*50)
        Logger.info(f"Fold: {fold}")
        
      
    best_f1 = 0    
    best_threshold = 0
    
    for t in range(1,100,1):
        t = t/100
        Precision = []
        Recall = []
        F1_score = []
        for fold in range(5):
            pred = (preds[fold]>=t).astype(np.int64)
                                
            precision = precision_score(y_vailds[fold], pred)
            recall = recall_score(y_vailds[fold], pred)
            f1 = f1_score(y_vailds[fold], pred)
            Precision.append(precision)
            Recall.append(recall)
            F1_score.append(f1)
        
        if sum(F1_score) / len(F1_score)>best_f1:
            best_f1 = sum(F1_score) / len(F1_score)
            best_Precision = Precision
            best_Recall = Recall
            best_F1_score = F1_score
            best_threshold = t
       
    if CFG.predict:
        for fold in range(5):
            submission = pd.read_csv("../外部資料集/38_Private_Test_Set_and_Submission_Template_V2/submission_template_public_and_private.csv")
            estimator = estimators[fold]
           
            predict = estimator.predict_proba(X_test)[:, 1]
            predict = (predict>=best_threshold).astype(np.int64)
            submission["飆股"] = predict
            submission.to_csv(f'./submission/{exp_id}_{fold}.csv',index=False)
            del predict 
    
        
    Logger.info("="*50)
    Logger.info(f"Precision: {best_Precision}")
    Logger.info(f"Recall: {best_Recall}")
    Logger.info(f"F1 score: {best_F1_score}") 
    Logger.info(f"{sum(best_Precision) / len(best_Precision)} {sum(best_Recall) / len(best_Recall)} {sum(best_F1_score) / len(best_F1_score)} {best_threshold}")
    
     
    Logger.info(f"Cost {time.time()-start} s")
    
    
    for handler in Logger.handlers[:]:
        Logger.removeHandler(handler)

