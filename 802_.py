import pandas as pd
import numpy as np

df=pd.read_csv('data/mpg.csv')
df1=df.iloc[3:10,2:7]
df2=df.iloc[100:107,1:4]
df2=df2.rename(columns={"year":"년도", "model":"모델"})
#
df2.shape
df1.shape
#
df1.iloc[4:6,2]
df1.loc[4:6,"cyl"]
df2
#
pd.merge(df1,df2,how="left",left_on="year",right_on="년도")


###
plt.clf()
import matplotlib.pyplot as plt    
x=np.linspace(-10, 10, 100)
y= (x ** 3) + 4*(x ** 2) + 6
plt.plot(x, y, color ="blue")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()

# 8_02 수업 
# ====================
# =====   옵션   =====
# ====================

import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
#def my_f(x):
    return x[0]**2 + x[1]**2 +3

#my_f([3,4])
##
# 최소값을 찾을 다변수 함수 정의
def my_f2(x):
    return (x[0]-1)**2 +(x[1]-2)**2 +(x[2]-4)**2 +7 

my_f([2,3,6])


# 초기 추정값
initial_guess = [0, 4,5]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

### kaggle house 이용하기
from sklearn.linear_model import LinearRegression
# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) #자동으로 기울기, 절편 값을 구해줌 

# 회귀 직선의 기울기와 절편
model.coef_ #기울기
model.intercept_ #절편

slope = model.coef_[0] 
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)
#
test=pd.read_csv('data/house/test.csv')
test_x=np.array(test["BedroomAbvGr"]).reshape(-1,1)

test_y=model.predict(test_x)
#
sub_df["SalePrice"] pred_y*1000





### copy ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


train =  pd.read_csv('data/house/train.csv')
# 트레인 데이터
x = np.array(train['GarageCars']).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array(train['SalePrice'])  # y 벡터 (레이블 벡터는 1차원 배열입니다)
# 선형 회귀 모델 생성
model = LinearRegression()
# 모델 학습
model.fit(x, y)               # 자동으로 기울기, 절편 값을 구해줌
# 회귀 직선의 기울기와 절편
model.coef_                   # 기울기 a
model.intercept_              # 절편 b
slope = model.coef_[0]
intercept = model.intercept_
# 예측값 계산
y_pred = model.predict(x)

test =  pd.read_csv('data/house/test.csv')
#test['Neighborhood'].value_counts()
#=df['Neighborhood']
test_x = np.array(test['GrLivArea']).reshape(-1, 1)
test.isnull().sum().sum()
test.fillna(0, inplace=True)
y_pred_hat = model.predict(test_x)
sub_df = pd.DataFrame({'Id' : test['Id'],
                       'SalePrice' : test['Id']})
sub_df['SalePrice'] = y_pred_hat
sub_df

sub_df.to_csv('sample_submission6.csv', index = False)

### gpt
import pandas as pd

# 예시 데이터프레임 생성
train1=train[["OverallQual",'SalePrice']]
x=train["OverallQual"]
y=train['SalePrice']
plt.scatter(x=x, y=y, s=3, alpha=0.5, color="black")
plt.show()
train1.corr()
train2["OverallQual"]=train2.index
train2=train.groupby("OverallQual")[["SalePrice"]].agg(["mean","median"])
#plt.scatter(data= train2, x=train2["OverallQual"],y=train2["SalePrice"], color="red")


## 이상치 탐색
train=train.sort_values(by='GrLivArea').iloc[:-2,:]
train.tail()
##
