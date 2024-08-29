import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x": x, "y": y
})

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

myindex=np.random.choice(30,30,replace=False)

train_set1=np.array([myindex[0:10], myindex[10:20]])
valid_set1=myindex[20:30]


def make_tr_val(fold_num, df, cv_num=3):
    np.random.seed(2024)
    myindex=np.random.choice(30,30,replace=False)
    #valid_index
    val_index=myindex[fold_num*10:(fold_num*10 +10)]

    #valid set, train set
    valid_set=df.loc[val_index]
    train_set=df.drop(val_index)

    train_X=train_set.iloc[:,1:] # X만 대문자인 이유: 대문자는 행렬, 소문자 벡터 
    train_y=train_set.iloc[:,0]

    valid_X=valid_set.iloc[:,1:]
    valid_y=valid_set.iloc[:,0]

    return (train_X, train_y,valid_X,valid_y)

train_X,train_y,valid_X,valid_y=make_tr_val(fold_num=2,df=df)
#tuple: 괄호를 안 써도 인식됨. 

from sklearn.linear_model import Lasso
val_result_total=np.repeat(0.0,3000).reshape(3,-1)
tr_result_total=np.repeat(0.0,3000).reshape(3,-1)


for j in np.arange(0, 3):
    train_X,train_y,valid_X,valid_y=make_tr_val(fold_num=j,df=df)

    # 결과 받기 위한 벡터 만들기
    val_result=np.repeat(0.0,1000)
    tr_result=np.repeat(0.0,1000)

    for i in np.arange(0,1000):
        model= Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)
    # model 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val
    
    val_result_total[j,:]=val_result
    tr_result_total[j,:]=tr_result

#산점도 그리기
import seaborn as sns

df=pd.DataFrame({
    "lambda": np.arange(0,10,0.01),
    "tr": tr_result_total.mean(axis=0),
    "val": val_result_total.mean(axis=0)
})

sns.scatterplot(data=df, x="lambda",y="tr") 
sns.scatterplot(data=df, x="lambda", y= "val", color="red") # red- valid set
plt.xlim(0,10)



# alpha  를 0.03 으로 선택 
np.argmin(val_result_total.mean(axis=0))
np.arange(0, 1, 0.01)[np.argmin(val_result_total.mean(axis=0))]
