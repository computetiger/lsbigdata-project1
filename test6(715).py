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
!pip install openpyxl
df_exam=pd.read_excel('data/excel_exam.xlsx') # sheet_name="Sheet2"
df_exam
sum(df_exam.math)/20
sum(df_exam.english)/20
sum(df_exam.science)/20
df_exam.shape
len(df_exam)
df_exam.size

df_exam['total']=df_exam['math']+df_exam['english']+df_exam['math']
df_exam['mean']=df_exam['total']/3

df_exam[(df_exam['math']>=50)&(df_exam['english']>=50)]


df_exam[(df_exam['math']>df_exam['math'].mean())&(df_exam['english']<df_exam['english'].mean())]
df_exam[df_exam['nclass']==3][['math','english','science']]
df_exam.iloc[:,3]

df_exam[1:2]
df_exam[1]
dd=df_exam.loc[1].to_frame()
type(dd)

df_exam.sort_values(['math','english'],ascending=False)
####
a=np.array([1,4,8,9,3,5,7,2,6,8])
np.where(a>3, "Up","Down")
####

df_exam["Updown"]=np.where((df_exam['math']>=50),"Up","Down")
df_exam.head()

#### 7.17
test1=pd.DataFrame({'id' :[1,2,3,40,5],
                    'midterm' : [60,80,70,90,85]})
test2=pd.DataFrame({'id' :[1,2,3,4,5],
                    'final' : [70,83,65,95,80]})

test1
test2

## merge167p
total=pd.merge(test1,test2,how='outer',on="id")
total
#
total=pd.merge(test1,test2,how='right',on="id")
total=pd.merge(test1,test2,how='inner',on="id")
total=pd.merge(test1,test2,how='left',on="id")

##169p
name=pd.DataFrame({'nclass': [1,2,3,4,5],
                    'teacher': ['kim','lee','park','choi','jung']})
df_exam
tot=pd.merge(df_exam,name,how="left",on="nclass")

## concat
group1=pd.DataFrame({'id' :[6,7,8,9,10],
                    'test' : [60,80,70,90,85]})
group2=pd.DataFrame({'id' :[6,7,8,9,10],
                    'test' : [70,83,65,95,80]})
group=pd.concat([group1,group2],axis=1)
