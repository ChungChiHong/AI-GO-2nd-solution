

import pandas as pd
import numpy as np

model1 = [14]
model2 = [17]

if len(model1)>0:
    print("model1 =",model1)
    vote1 = []
    for exp_id in model1:
        for i in range(5):
            df = pd.read_csv(f"submission/{exp_id}_{i}.csv")
            p = np.array(df["飆股"])
            vote1.append(p)
    vote1 = np.mean(np.array(vote1),0)
    vote1 = (vote1>0.5).astype(np.int64)
    print(sum(vote1[:25108]),sum(vote1[25108:]))

if len(model2)>0:
    print("model2 =",model2)
    vote2 = []
    for exp_id in model2:
        for i in range(5):
            df = pd.read_csv(f"submission/{exp_id}_{i}.csv")
            p = np.array(df["飆股"])
            vote2.append(p)
    vote2 = np.mean(np.array(vote2),0)
    vote2 = (vote2>0.0).astype(np.int64)
    print(sum(vote2[:25108]),sum(vote2[25108:]))
    

vote = vote1+vote2
vote = (vote>0.).astype(np.int64)
df["飆股"] = vote
df.to_csv(f'submission/{"_".join(map(str,model1))}_{"_".join(map(str,model2))}_ensemble.csv',index=False)
print(sum(vote[:25108]),sum(vote[25108:]))