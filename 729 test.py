import pandas as pd
import numpy as np    

np.random.seed(20240729)
old_seat=np.arange(1,29)
new_seat=np.random.choice(old_seat, 28, replace=False)

result=pd.DataFrame(
    {"old_seat":old_seat,
    "new_seat":new_seat}
)

result.to_csv("result.csv")
pd.to_csv("result.csv")

method: 객체 자체가 가지고 있는 함수, 객체가 있어야 사용할 수 있음. 
import matplotlib.pyplot as plt    
from scipy.stats import *

# 점을 직선으로 이어서 표현 
k=np.linspace(0,8,100)
y=uniform.pdf(k, loc=2, scale=4)
plt.scatter(k, y, color="black")
plt.show()
plt.clf()
#
x=np.linspace(-8,8,80)
y=x**2
plt.plot(x, y, color="blue")
plt.xlim(-10,10)
plt.ylim(0,40)

plt.show()
### ex.2)
data=np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
# 
dm=data.mean()
norm.ppf(0.90, loc=dm, scale=6)
#
dm == x bar
n =16
std = 6
alpha = 0.1
#
z_005=norm.ppf(0.95, loc=0, scale=1)
z_005
dm+z_005*6/np.sqrt(16)
dm-z_005*6/np.sqrt(16)
#
#표본분산 : 확률변수를 주체로 한 개념 
x=norm.rvs(loc=3, scale=5, size=10)
x**2
np.mean(x**2)
#X 가 N(3, 5^2)를 따를 때, e(x^2) 의 값은? 
k=np.array([0,0,0,0,0,0,0,0,0,0,0,0])
np.mean((k-k**2)/(2*k))
# 몬테카를로 적분: 확률변수 기댓값을 구할 때, 표본을 많이 뽑은 후, 원하는 형태로 변형, 평균을 계산해서 
#기댓값을 구하는 방법. 
# 표본 10만개 추출해 s^2 를 구하기. 
np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=100000)
x.mean()
s_2=sum((x-x.mean())**2)/ (100000-1)
#n-1 vs n
x=norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)
