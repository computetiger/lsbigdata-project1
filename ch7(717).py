import pandas as pd
import numpy as np

df=pd.DataFrame({'sex':['M','F',np.nan,'M','F'],
                 'score': [5,4,3,4, np.nan]})

#### 질문
df['score'].mean()




df['score']+1
pd.isna(df).sum().sum()

#결측치 제거하기
df.dropna(subset="score") # score 변수의 결측치 제거
df.dropna(subset=["score","sex"]) # 여러 변수 결측치 제거 
df.dropna()

exam=pd.read_csv('data/exam.csv')
exam
# loc indexing (location)
exam.loc[[2,7,14],'math']=np.nan
exam.iloc[[2,7,4],2]=30
exam.head()

df.loc[df["score"]==3,'score']=4 # 맞는것
df[df['score']==3]['score']=4 #틀림, 반영이 안됨
type(df[df['score']==4]) #dataframe
a=df[df['score']==3]
a['score']=4 ### 오류가 뜨는데 반영은 됨. 
### 몰?루

exam
exam.loc[(exam['math']<50),'math']=50
#iloc : 로지컬이든, 아니든 숫자 벡터로 들어오면 값을 받는다. 
#iloc을 사용해서 조회하려면 무조건 숫자벡터가 나와야함함
exam.iloc[exam[exam['english']>=90].index,3] =1000 # 조회도 되고 반영도 됨. 
exam.iloc[np.where(exam['english']>=90)[0],3] =1000 #np.where도 튜플이라 꺼내오면 됨.
exam.iloc[np.array(exam['english']>=90),3] =1000 # 조회, 반영 가능.

type(np.where(exam['english']>=90)) #tuple
np.where(exam['english']>=90)[0] # array

exam.loc[(exam['math']<=50),"math"]="-"
# 수학 -값을 수학 평균 값으로 대치 
exam.loc[exam['math']=='-','math']=pd.to_numeric(exam['math'],errors='coerce').mean()
exam.loc[exam['math']!='-','math'].mean() #- 가 아닌 것을 평균으로 할당. 
exam.loc[exam['math']=='-','math']=exam.query('math not in "-"').math.mean()
exam['math']=np.where(exam['math']='-',np.nan,exam['math'])

### np.array 사용하기
vector =np.array([np.nan if x=='-' else float(x) for x in exam['math']])
np.nanmean() # nan 을 제외하고 평균을 구함. 

math_mean=exam[exam['math']!= '-']['math'].mean()
exam['math']=exam['math'].replace("-",math_mean)  ##옳은 방법
