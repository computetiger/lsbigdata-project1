import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 불러오기
train = pd.read_csv('data/house/train.csv')
test  = pd.read_csv('data/house/test.csv')
sub   = pd.read_csv('sample_submission6.csv')

# 선형 회귀 모델
model = LinearRegression()
 
# 모델 학습
model.fit(x, y)               # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_                   # 기울기 a
model.intercept_              # 절편 b
slope = model.coef_[0]
intercept = model.intercept_

# 학습 데이터 선택하기
# 숫자형 변수만 선택하기
x =train.select_dtypes(include=[int,float])
x.info()
x.isnull().sum()
 
# 학습 x,y 선택
x = x.iloc[:,1:-1]
y = train["SalePrice"]
# train x의 결측치 처리 과정 
x["LotFrontage"]= x["LotFrontage"].fillna((x["LotFrontage"].mean()+x["LotFrontage"].median())/2)
x['GarageYrBlt']= x["GarageYrBlt"].fillna((x["GarageYrBlt"].mean()+x["GarageYrBlt"].median())/2)
x['MasVnrArea'] = x["MasVnrArea"].fillna((x["MasVnrArea"].mean()+x["MasVnrArea"].median())/2)

#결측치 확인
x.isnull().sum().sum()


#test 데이터 x, y 분리
test_x =test.select_dtypes(include=[int,float])
test_x=test_x.iloc[:,1:]
test_x.info()

# 평가 x 의 결측치 확인
test_x.isnull().sum()

# 함수 정의
new=[]
def change_null(x):
    test_x[x]=test_x[x].fillna((test_x[x].mean()+test_x[x].median())/2)
    new.append(test_x[x])
    return new

change_null(["LotFrontage","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageCars","GarageArea"])

# 평가 x의 결측치 확인
test_x.isnull().sum().sum()

# 회귀분석 학습하기
pred_y_hat=model.predict(test_x)
sub["SalePrice"]=pred_y_hat

sub.to_csv('sample_submission.csv', index = False)
