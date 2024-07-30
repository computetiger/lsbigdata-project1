!pip install pandas
!pip install numpy
import pandas as pd
import numpy as np

train=pd.read_csv('data/house/train.csv')
train_mean=train.groupby("YearBuilt")[["SalePrice"]].mean()

train[["Id","YearBuilt"]]

test=pd.read_csv('data/house/test.csv')
test=test[["Id", "YearBuilt"]]

new=pd.merge(test,train_mean,on="YearBuilt", how="left")

#결측치 전처리
cond1 =new["SalePrice"].isnull()
cond2 =(new["YearBuilt"]>=1900)
cond3 = (new["YearBuilt"]<1900)
new.loc[cond1&cond2, "SalePrice"] =  new.loc[cond2,"SalePrice"].mean()
new.loc[cond1&cond3, "SalePrice"] =  new.loc[cond3,"SalePrice"].mean()

new=new[["Id","SalePrice"]]

new.info()
new["Id"]=new["Id"].astype(str)
new.sort_values(by="Id", ascending=False)
new.to_csv("sample_submission.csv", index=False)





new["SalePrice"].fillna(new["SalePrice"].mean(), inplace=True)

new["SalePrice"].describe()


new.isnull().sum()

