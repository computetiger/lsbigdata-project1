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

lasso=Lasso(alpha=0.015)
ridge=Ridge(alpha=0.015)

lasso.fit(train_x, train_y)
ridge.fit(train_x, train_y)


y1=lasso.predict(test)
y2=ridge.predict(test)
y3=(y1 + y2 )/ 2 
sub["yield"]=y3
sub.to_csv("sample_submission3.csv", index=False)


array([4260.08064562, 6038.80817666, 7176.52275197, ..., 6858.02774247,
       4390.92783107, 7244.51027876])


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
