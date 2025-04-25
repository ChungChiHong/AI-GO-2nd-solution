

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
  
    model1 = 14  # 填入model1 id
    model1_threshold = 0.02  # 填入model1 best threshold
    
    predict = 1
    # feature
    select_col = ['券商代號965', '券商代號1050', '券商代號1024', '券商代號1071', '券商代號1056', '券商代號1048', '券商代號1069']
    extra_feature = 1
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
    df = pd.read_parquet("../外部資料集/train.parquet", engine='pyarrow')
    if CFG.extra_feature:
        onehot = pd.read_parquet("../外部資料集/train_onehot.parquet", engine='pyarrow') 
        #onehot = (onehot>0).astype(int)
        df = pd.concat([df, onehot], axis=1)

    
    if CFG.select_col:
        select_col = CFG.select_col
        df = df[select_col+["飆股","ID"]]
      
    if CFG.predict:
        X_test = pd.read_parquet('../外部資料集/public.parquet', engine='pyarrow')
        if CFG.extra_feature:
            onehot = pd.read_parquet("../外部資料集/public_onehot.parquet", engine='pyarrow') 
            #onehot = (onehot>0).astype(int)
            X_test = pd.concat([X_test, onehot], axis=1)
        
        X_test_private = pd.read_parquet('../外部資料集/private.parquet', engine='pyarrow')
        if CFG.extra_feature:
            onehot = pd.read_parquet("../外部資料集/private_onehot.parquet", engine='pyarrow') 
            #onehot = (onehot>0).astype(int)
            X_test_private = pd.concat([X_test_private, onehot], axis=1)
            
            X_test = pd.concat([X_test, X_test_private], axis=0)
        X_test = X_test.drop(columns="ID")
        
        if CFG.select_col:
            X_test = X_test[select_col]
   
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

        # filter sample
        X_train_fold = X_train[X_train.fold!=fold].reset_index(drop=True).drop(columns=["fold"])
        y_train_fold = y_train[y_train.fold!=fold].reset_index(drop=True).drop(columns=["fold"])
        X_valid_fold = X_train[X_train.fold==fold].reset_index(drop=True).drop(columns=["fold"])
        y_valid_fold = y_train[y_train.fold==fold].reset_index(drop=True).drop(columns=["fold"])
        
        drop_id = []
        for i in range(5):
            model1_predict = pd.read_csv(f'./predict/{CFG.model1}_{i}.csv')
            id = model1_predict[(model1_predict["p"] >=CFG.model1_threshold)].ID.tolist()
            drop_id = drop_id + id
            
        X_train_fold = X_train_fold.loc[~X_train_fold["ID"].isin(drop_id)]
        y_train_fold = y_train_fold.loc[~y_train_fold["ID"].isin(drop_id)]
        
        
        X_train_fold = X_train_fold.reset_index(drop=True).drop(columns=["ID"])
        y_train_fold = y_train_fold.reset_index(drop=True).drop(columns=["ID"])

        valid_id = X_valid_fold[["ID"]]
        X_valid_fold = X_valid_fold.reset_index(drop=True).drop(columns=["ID"])
        y_valid_fold = y_valid_fold.reset_index(drop=True).drop(columns=["ID"])
        
        estimator.fit(X_train_fold, y_train_fold)
        dump(estimator, f'model/{exp_id}_{fold}.joblib')  
        y_pred_fold = estimator.predict_proba(X_valid_fold)[:, 1]

        valid_id["pred_y"] = y_pred_fold
        valid_id["y"] = y_valid_fold
        valid_id.loc[valid_id['ID'].isin(drop_id), 'pred_y'] = 1
        y_pred_fold = valid_id["pred_y"]
        
        preds.append(y_pred_fold)
        estimators.append(estimator)
        y_vailds.append(y_valid_fold)
        
        Logger.info("-"*50)
        Logger.info(f"Fold: {fold}")
        
 
        del X_train_fold,y_train_fold,X_valid_fold,y_valid_fold,y_pred_fold
        gc.collect()
        
    best_f1 = 0
    best_p = 0    
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

    Logger.info(f"Precision: {sum(best_Precision) / len(best_Precision)}")
    Logger.info(f"Recall: {sum(best_Recall) / len(best_Recall)}")
    Logger.info(f"F1 score: {sum(best_F1_score) / len(best_F1_score)}") 
    Logger.info(f"Best threshold: {best_threshold}")
 
    Logger.info(f"Cost {time.time()-start} s")
    

    for handler in Logger.handlers[:]:
        Logger.removeHandler(handler)


