import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["species", "bill_length_mm", "bill_depth_mm"]]
df_x=df[["bill_length_mm", "bill_depth_mm"]]
df_y=df["species"] # y가 숫자 벡터가 아니어도 가능(tree)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(criterion="entropy",random_state=42, max_depth=2)

param_grid={
    'max_depth': np.arange(8,20,1),
    'min_samples_split': np.arange(1,30,1)
}

# confusion_matrix

model.fit(df_x, df_y)
model.predict(df_x)


grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    refit=True
)

grid_search.fit(df_x, df_y)

grid_search.best_params_ 
grid_search.cv_results_
grid_search.best_score_ 
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42, 
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_x, df_y)

from sklearn import tree
tree.plot_tree(model)

)