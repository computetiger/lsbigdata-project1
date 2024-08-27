import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

# 
valid_df = df.loc[20:]

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y=valid_df["y"]


from sklearn.linear_model import Lasso

val_result=np.repeat(0.0,100)
tr_result=np.repeat(0.0,100)

for i in np.arange(0,100):
    model= Lasso(alpha=i*0.1)
    model.fit(train_x, train_y)
# model 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

#seaborn 을 사용하여 산점도 그리기
import seaborn as sns

df=pd.DataFrame({
    "1": np.arange(0,1,0.01),
    "tr": tr_result,
    "val": val_result
})

sns.scatterplot(data=df, x="1",y="tr") 
sns.scatterplot(data=df, x="1", y= "val", color="red") # red- valid set
plt.legend()
plt.xlim(0,0.4)

# alpha  를 0.03 으로 선택 
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]

model=Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_

sorted_train=train_x.sort_values("x")
reg_line=model.predict(sorted_train)


# 문제 #
model= Lasso(alpha=1)
model.fit(train_x, train_y)
# -4부터 4까지의 x에 대한 예측값 계산
x_pred = np.arange(-4, 4, 0.01)

x_pred_poly = np.column_stack([x_pred ** i for i in range(1, 21)])  # 상수항 추가하지 않음
y_pred = model.predict(x_pred_poly)

# 시각화
plt.scatter(valid_df["x"], valid_df["y"], color="blue", label="Validation Data")
plt.plot(x_pred, y_pred, color="red", label="Lasso Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# 문제| shuffle, 5개의 랜덤 set
train_x=train_x.sample(frac=1).reset_index(drop=True)
train_y=train_y.sample(frac=1).reset_index(drop=True)

train_x1=train_x.iloc[0:4,]
train_x2=train_x.iloc[4:8,]
train_x3=train_x.iloc[8:12,]
train_x4=train_x.iloc[12:16,]
train_x5=train_x.iloc[16:,]

train_y1=train_y.iloc[0:4,]
train_y2=train_y.iloc[4:8,]
train_y3=train_y.iloc[8:12,]
train_y4=train_y.iloc[12:16,]
train_y5=train_y.iloc[16:,]
#VALID
valid_x
valid_y
#예측
x_pred = np.arange(-4, 4, 0.01)

x_pred_poly = np.column_stack([x_pred ** i for i in range(1, 21)])  # 상수항 추가하지 않음
np.random.shuffle(x_pred_poly)

# 모델학습
model=Lasso(alpha=0.03)
model.fit(train_x1, train_y1)

y_pred = model.predict(x_pred_poly)

x_pred_poly=x_pred_poly.reshape(4,-1)
x_pred_poly0=x_pred_poly[0].reshape(-1,1)


plt.scatter(valid_df["x"], valid_df["y"], color="blue", label="Validation Data")
plt.plot(x_pred_poly[0], y_pred0, color="red", label="Lasso Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
