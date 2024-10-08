# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
'''
import os
print(os.getcwd())  # 현재 작업 디렉토리 출력


## 필요한 데이터 불러오기
house_train=pd.read_csv("lsbigdata-project1/data/house/train.csv")
house_test=pd.read_csv("lsbigdata-project1/data/house/test.csv")
sub_df=pd.read_csv("lsbigdata-project1/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 선형 회귀 모델 생성
# 라쏘는?
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.03)
param_grid={
    "alpha":[0.1, 1.0, 10.0, 100.0]
    }

# 릿지는? 
from sklearn.linear_model import Ridge
rd = Ridge(alpha=0.03)
lr = LinearRegression()

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestRegressor

ela = ElasticNet()
rfr=RandomForestRegressor(n_estimators=100) # n_ : tree 개수

param_grid={
    "alpha": [0.1 , 1.0 , 10.0 , 100.0], 
    "l1_ratio":[0, 0.1,0.5,1.0]}

grid_search = GridSearchCV(
    estimator=ela,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5
)

#그리드서치  for randomforests
param_grid={"max_depth":[3,5,7],
"min_samples_split":[20,10,5],
"min_samples_leaf":[5,10,20,30],
}

grid_search.fit(train_x, train_y)

grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

# 모델 학습
best_model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)