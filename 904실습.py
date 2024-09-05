import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(2024)
x=np.linspace(10,10,100)

y=x**2 + np.random.normal(0,10,size=x.shape)

df = pd.DataFrame({"x":x,"y":y})
plt.scatter(df.x, df.y )



x_train,x_test,y_train, y_test = train_test_split(df.x,df.y,test_size=0.3)

# 디시전 
model=DecisionTreeRegressor(random_state=42, max_depth=3) # 깊이 : 3 , 8개 
model.fit(x_train, y_train)
df_x=pd.DataFrame({"x":x})

y_pred=model.predict(df_x)

plt.scatter(df["x"],df["y"], label="Noisy Data")
plt.xlabel()
