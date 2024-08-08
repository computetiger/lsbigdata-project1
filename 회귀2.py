!pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([2, 3, 4]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([147.7235, 175.5005, 206.9605])  # y 벡터 (레이블 벡터는 1차원 배열입니다)


#x=np.array(train_mean["BedroomAbvGr"]).reshape(-1, 1)
#y=np.array(train_mean["SalePrice"]) 

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

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='LinearRegression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()
# house data
train=pd.read_csv("data/house/train.csv")
train_mean=train.groupby("BedroomAbvGr")[["SalePrice"]].mean().astype(int)
train_mean.value_counts()
