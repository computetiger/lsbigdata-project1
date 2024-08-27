# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 필요한 데이터 불러오기
house_train=pd.read_csv("data/house/train.csv")
house_test=pd.read_csv("data/house/test.csv")
sub_df=pd.read_csv("sample_submission10.csv")

# 계산의 편의를 위해 두 데이터 합치기 
df=pd.concat([house_train,house_test]).reset_index(drop=True)

# 네이버후드 더미화 
n_dummies=pd.get_dummies(df["Neighborhood"], drop_first=True)

#train data 와 네이버후드 더미 합쳐 train, test 셋
x=pd.concat([house_train[["GrLivArea","GarageArea"]],n_dummies],axis=1)
#y=df["SalePrice"]



#즉, train 을 trian, valid 로 나누기
train_n=len(house_train)

train=x.iloc[:train_n,]
test=x.iloc[train_n:,]
#
x_train=y[:train_n]
y_train=train


#x, y, 나누기
# ^ :  시작을 의미, $ 끝남을 의미, | = or 를 의미
# regex :  regular expression, 정규방정식.
train.filter(regex="^GrLivArea$|^GarageArea$|^Neighborhood")


# Validation Set 만들기 (모의고사 셋)
np.random.seed(42)
val_index=np.random.choice(np.arange(1460), size=438, replace=False)

x_valid=x_train.loc[val_index] # 30%
x_train=x_train.loc[~x_train.index.isin(val_index),:] # 70% , #x_train=x_train.drop(val_index) 
y_valid=y_train.loc[val_index] # 30%
y_train=y_train.drop(val_index) # 70%

#이상치 탐색
house_train=house_train.query("GrLivArea<= 4500")
train_n=house_train.shape[0]

# 선형회귀모델 생성
model=LinearRegression()
model.fit(x_train, y_train)

#성능 측정
y_hat=model.predict(x_valid)
np.sqrt(((y_valid-y_hat)**2).mean())

from sklearn.metrics import mean_squared_error
mean_squared_error(y_valid,y_hat)


