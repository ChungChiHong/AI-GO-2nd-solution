# 2025永豐AI GO競賽：股神對決 2nd solution
競賽：[2025永豐AI GO競賽：股神對決](https://tbrain.trendmicro.com.tw/Competitions/Details/38)
## Installation
```bash
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install scikit-learn==1.0.2
pip install xgboost==2.0.0
pip install imblearn
```

## Usage
### step1：
將38_Private_Test_Set_and_Submission_Template_V2、38_Public_Test_Set_and_Submmision_Template_V2、38_Training_Data_Set_V2競賽資料集放置於外部資料集下

外部資料集   
├─38_Private_Test_Set_and_Submission_Template_V2  
├─38_Public_Test_Set_and_Submmision_Template_V2  
├─38_Training_Data_Set_V2  
└─columns.csv  

執行csv2parquest.py將csv轉為parquest，程式會自動將parquest存於外部資料集內
```bash
cd 程式碼檔案資料夾路徑
python csv2parquest.py
```
執行成功後，外部資料集內容應如下：

外部資料集                     
├─38_Private_Test_Set_and_Submission_Template_V2  
├─38_Public_Test_Set_and_Submmision_Template_V2  
├─38_Training_Data_Set_V2   
├─private.parquet  
├─public.parquet  
├─train.parquet  
└─columns.csv  

### step2：
執行train1.py，訓練model1
```bash
cd 程式碼檔案資料夾路徑
python train1.py
```
執行成功後，model1結果應顯示：  

**Precision**: 1.0  
**Recall**: 0.7224489795918367  
**F1 score**: 0.8385839276382583  
**Best threshold**: 0.02  

### step3：
執行onehot.py，程式會自動將parquest存於外部資料集內
```bash
cd 程式碼檔案資料夾路徑
python onehot.py
```
執行成功後，外部資料集內容應如下：

外部資料集                     
├─38_Private_Test_Set_and_Submission_Template_V2  
├─38_Public_Test_Set_and_Submmision_Template_V2  
├─38_Training_Data_Set_V2  
├─private.parquet  
├─public.parquet  
├─train.parquet  
├─private_onehot.parquet  
├─public_onehot.parquet  
├─train_onehot.parquet  
└─columns.csv  
### step4：
將model1 id 填入CFG，執行train2.py，訓練model2
```python
class CFG:
    model1 = 1  # 填入model1 id
    model1_threshold = 0.02  # 填入model1 best threshold
```
```bash
cd 程式碼檔案資料夾路徑
python train2.py
```
執行成功後，model2結果應顯示：  

**Precision**: 1.0  
**Recall**: 0.9482993197278912  
**F1 score**: 0.9734493745858888  
**Best threshold**: 0.01 

### step5：
將model1、model2 id 填入CFG，執行ensemble.py
```python
class CFG:
    model1 = [int]  # 填入model1 id
    model2 = [int]  # 填入model2 id
```
```bash
cd 程式碼檔案資料夾路徑
python ensemble.py
```
執行成功後，會將model1與model2結果ensemble並儲存於submission資料夾內，檔名為{model1 id}_{model2 id}_ensemble.csv。  

**ensemble結果應在public中預測出167個正樣本，在private中預測出164個正樣本**

### other：
find_importance_feature.py 用於列出重要特徵，並儲存於importance資料夾內
```python
class CFG:
    exp_id = 16  # 填入model id
    importance_type = "gain"  # weight or gain
```


