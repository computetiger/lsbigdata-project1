# y=2x+3 그래프 그리기 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

plt.clf()
# x 값 설정
x=np.linspace(0,100,400)
# 그래프 설정
y=2*x+3
plt.plot(x,y,color="black")

# scatter 생성
obs_x=np.random.choice(np.arange(100),20)
obs_y=2*obs_x+3+epsilon_i
epsilon_i=norm.rvs(loc=0, scale=40, size=20)

plt.scatter(obs_x,obs_y, color="red",s=3)

# df로 만들기 
df=pd.DataFrame({
    "x": obs_x,
    "y": obs_y})
df

#회귀분석
from sklearn.linear_model import LinearRegression
model = LinearRegression()

obs_x=obs_x.reshape(-1,1)
model.fit(obs_x, obs_y) 

# 기울기와 절편 
model.coef_   #기울기  a hat
model.intercept_  # 절편  b hat

x_1=np.linspace(0,100,400)
y_1=model.coef_*x_1 + model.intercept_
plt.plot(x_1,y_1, color="Blue")
plt.xlim([0,100])
plt.ylim([0,300])
plt.show()

# chap. 2

#!pip install statsmodels
import statsmodels.api as sm

obs_x=sm.add_constant(obs_x)
model=sm.OLS(obs_y, obs_x).fit()
print(model.summary())

'''
np.sqrt((8.79)**2/20)*3 + 10
1 - norm.cdf(18, loc=10, scale=1.96) 
#0.0002 정도의 확률 >> 귀무가설 기각
'''
