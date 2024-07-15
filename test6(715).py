import pandas as pd
import numpy as np

df=pd.DataFrame({'name':['김지훈','이유진','박동현','김민지'],
                 'english' :[90,80,60,70],
                 'math' : [50,60,100,20]})
                 
type(df.name)
type(df.math)
df
df.name
df[['name']]
#Q1
df=pd.DataFrame({'제품':['사과','딸기','수박'],
                 '가격' :[1800,1500,3000],
                 '판매량' : [24,38,13]})
print(f"과일의 가격 평균: {df['가격'].mean()}, 판매량의 평균: {df["판매량"].mean()} ")

#
df_exam=pd.read_excel('excel_exam.xlsx')
df_exam
