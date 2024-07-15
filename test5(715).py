#07.15
#브로드캐스팅 
import pandas as pd
import numpy as np    
a=np.array([1.0,2.0,3.0])
b=2.0
a*b

a.shape #(3, )
b.shape #(어트리뷰트 에러)

#2차원 배열
matrix=np.array([[0,0,0],[10,10,10],[20,20,20],[30,30,30]]) 
matrix.shape

vector=np.array([1,2,3])
vector.shape

result=matrix +vector 
result

#
vec=np.array([1,2,3,4]).reshape(4,1)
result=matrix+vec
result

#
np.random.seed(2024) #랜덤 값을 고정해준다. 
a=np.random.randint(1,21,10)
print(a)

#
a[2:5]
a[:2]
#1부터 1000사이 3의 배수의 합은? 
sum(np.arange(0,1000,3)) #1번 방법
a=np.arange(0,100) # 2번 방법
sum(a[::3])
#
np.delete(a,[1,2,4])
#불리언 인덱싱을 사용한 필터링 작업
a[a>50]

#
np.random.seed(2024)
a=np.random.randint(1,10000,300)
a[(a>200)&(a<5000)]

#
!pip install pydataset
import pydataset
df= pydataset.data('mtcars')
np_df=np.array(df['mpg']) #배열 1
df_mpg=list(df.mpg) #배열 2

#mpg 중 15이상 25이하인 데이터의 개수는 ?
df.mpg[(df.mpg>=15)&(df.mpg<=25)].count() #1번 방법
sum((df.mpg>=15)&(df.mpg<=25)) #2번 방법

#평균 mpg보다 높은 자동차 대수?
sum((df.mpg.mean()<=df.mpg))

#
np.random.randint(2024)
a=np.random.randint(1,10000,5)
b=np.array(['A','B','C','F','H'])

a[((a>2000)&(a<5000))]
b[((a>2000)&(a<5000))]

model_names=np.array(df.index)
#model_names[np.mean(np.mpg)<=np.mpg]
a[a>5000]=30000
a

#
import numpy as np
a=np.array([1,3,8,4,9])

result=np.where(a<6)

#처음으로 5000보다 큰 숫자가 나왔을 떄 숫자 위치와 그 숫자 값
np.random.seed(2024)
b=np.random.randint(1,26356,1000)
b
c=np.where(b>5000)

b[b>10000][0]

x=np.where(b>22000)
type(x)
y=x[0][0]
z=b[y]
y, z

# 
np.random.seed(2024)
b=np.random.randint(1,26356,1000)
x=np.where(b>24000)
x[0][0]
#
x=np.where(b>10000)
type(x[0][0])
y=x[0][49]
z=b[y]
print(f"숫자의 위치는{y},숫자는{z}")
#
x=np.where(b<500)
y=x[0][-1]
z=b[y]
print(f"숫자의 위치는{y},숫자는{z}")

#np.nan
import numpy as np
a=np.array([20,3,53,8,np.nan, 6])
np.mean(a)
np.nanmean(a)
np.nan_to_num(a,nan=0)

#
a=True
b=np.nan
b
a

# 다시보기
a+1
b+1
a_filtered=b[~np.isna(b)]
# 문자와 숫자 섞어서 벡터 생성성
import numpy as np
str_vec = np.array(["사과", "배", "딸기", "참외"])
str_vec
str_vec[[0,2]]

import numpy as np
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec
#concat은 튜플로 들어오건, 리스트로 들어오건 무조건 묶어준다. 
combined_vec = np.concatenate([str_vec, mix_vec])
type(combined_vec)

# 벡터들을 세로로 붙여줍니다. 
np.arange(1,5)
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked
#
row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

#
import numpy as np
# 길이가 다른 벡터
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
np.resize(vec1,len(vec2))
vec1 = np.resize(vec1, len(vec2))
vec1

#
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
a=np.vstack((vec1,vec2))
#
a = np.array([12, 21, 35, 48, 5])
a[::2]
#
a=np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)
#
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
combined_vec = np.concatenate([a, b])

vec1=np.empty(6)
vec1[[0,2,4]]=a
vec1[[1,3,5]]=b
vec1
