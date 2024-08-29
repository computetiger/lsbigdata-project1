# 필요한 패키지 불러오기
# 모든 변수를 lasso R 사용해 분석하기. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso

## 필요한 데이터 불러오기
house_train=pd.read_csv("data/house/train.csv")
house_test=pd.read_csv("data/house/test.csv")
sub_df=pd.read_csv("sample_submission10.csv")

# 각 숫자변수는 평균 채우기 
num_type=house_train.select_dtypes(include="number")
num_type=num_type.fillna(num_type.mean())
house_train=num_type

train_n=len(house_train)

# train / test 데이터셋
train_x=house_train.iloc[:,1:-1]
train_y=house_train.iloc[:,-1]


# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model,train_x, train_y, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 400, 0.1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)
####

test_x=house_test.drop("SalePrice", axis=1)
# 선형 회귀 모델 생성
model = Lasso(alpha=optimal_alpha)

## submission
df_test_num = house_test.select_dtypes(include=['number'])
df_test_num = df_test_num.fillna(df_train_num.mean())

X_test_num = df_test_num.iloc[:, 1:]
y_test_pred = lasso.predict(X_test_num)

df_sub['SalePrice'] = y_test_pred

df_sub.to_csv("sample_submission12.csv", index=False)