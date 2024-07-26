import pandas as pd
import numpy as np

#데이터 전처리 함수 
#query 
#df[]
#sort_values()
#groupby()
#assign()
#agg()
#merge()
#concat()

df=pd.read_csv('data/exam.csv')
df
#query : 조건에 맞는 행을 걸러내는 작업
df.query("nclass==1")
df.query("nclass==2")
df.query("nclass!=3")
df.query('math>50')
df.query("english<= 80")
df.query("nclass==1 & math>=50")
df.query("math>= 50 | english >= 90")
df.query("nclass==1 | nclass==2 | nclass==4")
df.query("nclass not in [1,3,5]")
#  df[~df["nclass"].isin([1,2])]

nclass1=df.query("nclass==1")
nclass1.math.mean()
#
df['nclass']
df[['id','nclass']]
df.drop(columns=["math", "english"])
df.query("nclass==1")[["math","english"]]
df.sort_values("math",ascending=False)

# 변수추가 
df=df.assign(total =df["math"] + df["english"]+ df["science"])
df=df.assign(mean=df['total']/3)
df
# 그룹을 나눠 요약함
df.agg(mean_math=("math","mean"))
df.groupby("nclass")\
    .agg(mean_math=("math","mean"))

## 질문 
df.groupby("nclass") \
    .agg(mean_math=("math","mean"),
    mean_sci=("science","mean"),
    mean_eng=("english","mean"))

####    
import pydataset 
mpg=pydataset.data("mpg")
mpg.groupby(["manufacturer","drv"]).agg(mean_cty=("cty","mean"))

mpg.query("manufacturer=='audi'").groupby("drv").agg(n=("drv","count"))
