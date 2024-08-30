import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform


from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")
sub=pd.read_csv("data/sample_submission.csv")

train_x=train.iloc[:,1:-1]
train_y=train["yield"]

plt.boxplot(train_x)

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

ridge=Ridge(alpha=10)
lr=LinearRegression()
#dtr= DecisionTreeRegressor(max_depth=10)

hard=VotingRegressor(
    estimators=[("lasso",lasso),("ridge",ridge), ("lr",lr)])

hard.fit(train_x, train_y)

print(hard.score(train_x, train_y))

y=hard.predict(test)
len(y)

sub["yield"]=y
sub.to_csv("sample_submission1.csv", index=False)

##-------------------------------최적화 :하이퍼 파라미터 찾아보기
from sklearn.neighbors import KNeighborsRegressor

k_range=range(1,101)
train_score=[]
test_score=[]

for i in k_range:
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(train_x, train_y)
    test_p=knn.predict(test)

    train_score.append(knn.score(train_x, train_y))
    test_score.append(knn.score(test, test_p))

tmax=train_score.index(max(train_score))
tmin=train_score.index(min(train_score))

knn=KNeighborsRegressor(n_neighbors=3)
knn.fit(train_x, train_y)
test_p=knn.predict(test)
len(test_p)

sub["yield"]=test_p
su
sub.to_csv("sample_submission1.csv", index=False)

train

import numpy as np
a=np.array([1,1,5,7,8,8])
np.std(b)
np.sqrt(180)* 0.37
1.87 * 20
b=np.array([7,7,7,-2,-2])
#________________________ 남규
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

## 필요한 데이터 불러오기
berry_train=pd.read_csv("data/train.csv")
berry_test=pd.read_csv("data/test.csv")
sub_df=pd.read_csv("sample_submission.csv")

berry_train.isna().sum()
berry_test.isna().sum()

berry_train.info()

## train
X=berry_train.drop(["yield", "id"], axis=1)
y=berry_train["yield"]
berry_test=berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled=scaler.transform(berry_test)

# 정규화된 데이터를 DataFrame으로 변환
X = pd.DataFrame(X_scaled, columns=X.columns)
test_X= pd.DataFrame(test_X_scaled, columns=berry_test.columns)

polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(X.values)
features=polynomial_transformer.get_feature_names_out(X.columns)
X=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_X.values)
features=polynomial_transformer.get_feature_names_out(test_X.columns)
test_X=pd.DataFrame(polynomial_features,columns=features)

#######alpha
# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(2, 4, 1)
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

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()


### model
model= Lasso(alpha=2.9)

# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y_lasso=model.predict(test_X) # test 셋에 대한 집값
pred_y





# ridge==========================================================================
# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(70, 80, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)






import pandas as pd
import numpy as np
import matplotlib as plt
skl