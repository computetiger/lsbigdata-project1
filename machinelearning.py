import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"]=iris.target
X=df.iloc[:,:-1]
Y=df["target"]
Y

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
x_train, x_test, y_train, y_test=train_test_split(X, Y, random_state=0, test_size=0.25)

model=LinearRegression()
model.fit(x_train, y_train)

model.score(x_train, y_train) , model.score(x_test, y_test)
y_test
p_test=model.predict(x_test)


classification_report(y_test, p_test)