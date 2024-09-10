# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from palmerpenguins import load_penguins
pen

df=sns.load_dataset("penguins")

df.isnull().sum()
df.iloc[:,2:6]=df.iloc[:,2:6].fillna(df.iloc[:,2:6].mean())
df.sex=df.sex.fillna("Male")
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

X=df.drop(columns="bill_length_mm")
Y=df.bill_length_mm


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(X, Y, test_size=0.3)


# ElasticNet / 그리드서치
from sklearn.linear_model import ElasticNet
ela = ElasticNet()
param_grid={
    "alpha": [0.1 , 1.0 , 10.0 , 100.0], 
    "l1_ratio":[0, 0.1,0.5,1.0]}

from sklearn.model_selection import  GridSearchCV
grid_search = GridSearchCV(
    estimator=ela,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3, 
    refit=True
)


grid_search.fit(x_train, y_train)

grid_search.best_params_ # 0.1,1
grid_search.cv_results_
grid_search.best_score_# -5.9
best_model=grid_search.best_estimator_
#-----------------------------------------
#  DecisionRegressor/ 그리드서치
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=2024)
param_grid={
    'max_depth': np.arange(8,20,1),
    'min_samples_split': np.arange(1,30,1)
}

grid_search=GridSearchCV(
    estimator=dtr,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    refit=True
)

grid_search.fit(x_train, y_train)

grid_search.best_params_ #max_depth:50, mss:10
grid_search.cv_results_
grid_search.best_score_ #-7.7
best_model=grid_search.best_estimator_

model = DecisionTreeRegressor(random_state=42, 
                              max_depth=8,
                              min_samples_split=22)
model.fit(x_train, y_train)

from sklearn import tree
tree.plot_tree(model)