import pandas as pd
import numpy as np
df=pd.read_csv('data/project.csv')
df

# 변환계수 설정, 2011년 기준으로 변환
num=df['cpi'][0]/df['cpi'][9]
df['cpi']=df['cpi']/num
df.head()

# 변수 이름 변경
df.rename(columns={'cpi':'2011_cpi'})

#필터링
df_fil=df[df['year']>=2021]
df_fil

#CPI 변화율, 인플레이션율
num2=(df['cpi'][1]-df['cpi'][0]) /df['cpi'][0]*100

def infl(x) :
    a=df['cpi'][2]-df['cpi'][1]) /df['cpi'][1]*100
    return a
df['infl']=(df['cpi'][1]-df['cpi'][0]) /df['cpi'][0]*100


##ex. 
df=pd.read_csv('data/project1')
df.head(3)

