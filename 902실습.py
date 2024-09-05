import numpy as np
import pandas as pd
x=[145, 78]
y=[300, 200]
z_score, p_vlaue = proportions_ztest(count=x, nobs=n)
mat_b=np.array([[174, 124], [193,107],[159,141]])
chi2, p, df, expected = chi2_contingency(mat_b, correction=)
chi2.round(3)  #검정 통계량
p.round(4) # p-value
# 904 실습

def my_mse(x):
    n1=df.query(f"x< {x}").shape[0]
    n2=df.query(f"x>={x}").shape[0]
    y_hat1=df.query(f"x<{x}").mean()[0]
    y_hat2=df.query(f"x>={x}").mean()[0]
    mse1=np.mean((df.query(f"x < {x}")["y"]- y_hat1)** 2)
    mse2=np.mean((df.query(f"x >= {x}")["y"]- y_hat2)** 2)
    return (mse1 * n1 + mse2 * n2) / (n1 + n2)

my_mse(20)

df["x"].min()
df["x"].max()

from scipy.optimize import minimize
# 초기 추정값
initial_guess=13.5

result = minimize(my_mse, initial_guess)
x_values=np.linspace(start=13.2, stop = 21.4, num=100)
result=np.repeat(0,100)
for i in range(100):
    result[i]=my_mse(x_values[i])

my_mse(x_values[0])

np.argmin(result)

# MSE : 
result = minimize(my_mse, initial_guess)
x_values=np.linspace(start=13.2, stop = 21.4, num=100)
result=np.repeat(0,100)
for i in range(100):
    result[i]=my_mse(x_values[i])

my_mse(x_values[0])



#13-22 사이 값 중 0.01 간격으로 MSE 계산을 해서 
#minimize 사용해서 가장 작은 MSE 가 나오는 x 찾아보세요!


import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
df = penguins.dropna()
df = df[["bill_length_mm", "bill_depth_mm"]]
df = df.rename(columns={"bill_length_mm": "y", "bill_depth_mm": "x"})

# 원래 MSE는??
((df["y"] - df["y"].mean()) ** 2).mean()

# MSE 계산 함수 정의
def MseByX(X, df):
    df1 = df.query(f'x < {X}')
    df2 = df.query(f'x >= {X}')
    
    n1 = len(df1)
    n2 = len(df2)

    yhat1 = df1["y"].mean()
    yhat2 = df2["y"].mean()

    MSE1 = ((df1["y"] - yhat1) ** 2).mean()
    MSE2 = ((df2["y"] - yhat2) ** 2).mean()
    return (MSE1 * n1 + MSE2 * n2) / (n1 + n2)


def find_split(df, step=0.01):
    x_list = np.arange(df["x"].min(), df["x"].max(), step)[1:-1]
    mse_list = [MseByX(x, df) for x in x_list]

    x = x_list[np.argmin(mse_list)]
    return x


#16.4 찾기
find_split(df, step=0.01)

df_left = df.query(f"x < {find_split(df, step=0.01)}")
df_right = df.query(f"x >= {find_split(df, step=0.01)}")

#좌측
find_split(df_left, step=0.01)

#우측
find_split(df_right, step=0.01)

df=df.query("x>=16.41")


x_values=np.arange(16.51, 21.5, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])
result
df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01,16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)  # digitize: 기준점이 있을 떄 기준을 중심으로,
df.groupby("group").mean()


df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01,16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)  # digitize: 기준점이 있을 떄 기준을 중심으로,
df.groupby("group").mean()

k1=np.linspace(10,14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k2=np.linspace(14.01, 100)
k2=np.linspace(19.axis=.01, 100)


