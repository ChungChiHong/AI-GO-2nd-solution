# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:52:14 2025

@author: foresight
"""

import pandas as pd
from joblib import load
import os


class CFG:
    
    exp_id = 16
    importance_type = "gain"  # weight  gain
    
if __name__ == "__main__":
    
    os.makedirs('importance', exist_ok=True)
    
    dfs = []
    for i in range(5):
        
        estimator = load(f'model/{CFG.exp_id}_{i}.joblib')
        
        xgb_model = estimator.named_estimators_['xgb'].named_steps['classifier']
        xgb_feature_importance = xgb_model.get_booster().get_score(importance_type=CFG.importance_type)
        importance = pd.DataFrame({
            'feature': xgb_feature_importance.keys(),
            'importance': xgb_feature_importance.values()
        }).sort_values(by='importance', ascending=False)
        
    
        dfs.append(importance)
    merged_df = pd.concat(dfs, ignore_index=True)
    df = merged_df.groupby('feature', as_index=False)['importance'].sum().sort_values(by='importance', ascending=False)
    
    df['importance'] = df['importance']/5
    print(df)
    
    df.to_csv(f"./importance/{CFG.exp_id}_{CFG.importance_type}.csv")
    
            
