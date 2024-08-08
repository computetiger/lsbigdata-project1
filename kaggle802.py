import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# 데이터 불러오기
train =  pd.read_csv('data/house/train.csv')
test =  pd.read_csv('data/house/test.csv')
sub =pd.read_csv('sample_submission6.csv')

# 이상치 탐색
#train["GarageArea"]

# 회귀분석 적합

x=train[['GrLivArea',"GarageArea"]]   # pandas Series
y=train["SalePrice"]
# 선형 회귀 모델
model = LinearRegression()
# 모델 학습
model.fit(x, y)               # 자동으로 기울기, 절편 값을 구해줌

#test_y = test["SalePrice"]

# 회귀 직선의 기울기와 절편
model.coef_                   # 기울기 a
model.intercept_              # 절편 b
slope = model.coef_[0]
intercept = model.intercept_


### 3교시
from scipy.optimize import minimize

def house_f(x,y):
    return model.coef_[0]*x + model.coef_[1]*y + model.intercept_

# Test data 만들기
test_x = test[['GrLivArea',"GarageArea"]] 
# 결측치 채우기
test_x.fillna(test_x["GarageArea"].mean(), inplace=True)

# 함수 적용
house_f(test_x['GrLivArea'], test_x["GarageArea"])
# 예측값 계산
pred_y = model.predict(test_x)
pred_y

sub["SalePrice"]=pred_y
sub.to_csv('sample_submission10.csv', index = False)

'''
# 초기 추정값
initial_guess = [0, 4]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
