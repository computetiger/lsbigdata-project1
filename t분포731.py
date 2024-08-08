import pandas as pd
import numpy as np    
from scipy.stats import *
import seaborn as sns    
import matplotlib.pyplot as plt    
# numpy의 분산과 pandas의 분산의 ddof default가 다르다 
data=np.array([1,3,5,7,9,6,8,20,50,52,26,74,31,16])
np.var(data) #default : n

data1=pd.DataFrame({
    "columns": [1,3,5,7,9,6,8,20,50,52,26,74,31,16]})
data1.var(ddof=0)  #default : n-1
data1.var()

x=norm.ppf(0.25, loc=3, scale=7)
z=norm.ppf(0.25, loc=0, scale=1)
z*7 + 3 # x 의 값과 같다. 
'''
x= z*7 +3
(x - 3) / 7 = z
(x - mu) / sigma = z
'''
norm.cdf(5, loc=3, scale=7)  # 0.612
norm.cdf(2/7, loc=0, scale=1) #0.612
'''
x bar +- 1.96 * s/ sqrt (n) 
'''
plt.clf()
x=norm.rvs(loc=0, scale=1, size=100000)
x_values=(np.linspace(x.min(), x.max(), 300))
pdf_values=norm.pdf(x_values, loc=0, scale=1)
plt.plot(x_values, pdf_values, color ="r")
sns.histplot(data=x,bins=40, stat="density")
plt.show()


from scipy.stats import norm
x = norm.ppf(0.25, 3, 7)
z = norm.ppf(0.25, 0, 1)

x
3 + z *7

##  x = 3 + z * 7
##  x - 3 = z * 7
##  z = 7 / (x - 3)
##  X - M / STD = Z 


norm.cdf(5, 3, 7)  ## x
norm.cdf(2/7, 0, 1) ## z

norm.ppf(0.975, 0, 1)  ##1.96
norm.ppf(0.025, 0, 1)  ##1.96


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


z = norm.rvs(0, 1, size = 1000)
x = ( z * np.sqrt(2) ) + 3


zmin = z.min()
zmax = z.max()

xmin = x.min()
xmax = x.max()

z_values = np.linspace(zmin, zmax, 100)
x_values = np.linspace(xmin, xmax, 100)

pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
pdf_values2 = norm.pdf(x_values, loc = 3, scale = np.sqrt(2))

plt.plot(z_values, pdf_values, color = 'red')
plt.plot(x_values, pdf_values2, color = 'red')

sns.histplot(z, stat='density') #stat = density = y축을 비율로 바꿈
sns.histplot(x, stat='density')

plt.show()
plt.clf() 
# 과제  X(5, 3^2)
# 1. X표본을 10개 뽑기
x_10 = norm.rvs(5, 3, size = 10)
#x_10 = np.linspace(zmin, zmax, 100)
# 2. X표본 1000개 뽑기
x_1000 = norm.rvs(5, 3, size = 1000)
# 3. 1번에서 계산한 s으로 > sigma 대체한 표준화를 진행
np.std(x_10, ddof=1)
z=(x_1000 -5) / 3
# 4. z 의 히스토그램 그리기 >> 표준정규분포 pdf
x_values = np.linspace(x_1000.min(), x_1000.max(), 1)

####
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)

plt.plot(z_values, pdf_values, color = 'red')
sns.histplot(z,stat="density")
plt.show()
plt.clf()


# t분포  : 종 모얀, 대칭, 중심 0 
# X ~ t(df) 
# 모수 df(n) :자유도라고 부름 - 퍼짐을 나타내는 모수, df(n) 이 작으면 분산 커짐
from scipy.stats import t
# 자유도가 4 인 t분포 
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color="red", linewidth=2)
plt.show()
#### t분포와 정규분포 동시에 표현 #####
plt.clf()
#정규분포
z = norm.rvs(0, 1, size = 1000)
z_values = np.linspace(z.min(), z.max(), 100)
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
plt.plot(z_values, pdf_values, color = 'green')

# t분포
t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color="red", linewidth=2)

# t분포 (자유도 : 8)
t_values2 = np.linspace(-4, 4, 100)
pdf_values2 = t.pdf(t_values2, df=8)
plt.plot(t_values2, pdf_values2, color="red", linewidth=2)

# t분포 (자유도 : 2)
t_values3 = np.linspace(-4, 4, 100)
pdf_values3 = t.pdf(t_values3, df=2)
plt.plot(t_values3, pdf_values3, color="orange", linewidth=2)

# t분포 (자유도 : 50)
t_values4 = np.linspace(-4, 4, 100)
pdf_values4 = t.pdf(t_values4, df=30)
plt.plot(t_values4, pdf_values4, color="red", linewidth=2)

plt.show()


## 정리 ##
# X ~ ?(mu, sigma^2)
# X bar ~ N ( mu, sigma ^ 2/ n)
# X bar ~= t( x_bar, s ^ 2/ n) 자유도가 ( n-1 ) 인 t분포

### df = degree of freedom 
### 모분산을 모를 때 : 모평균에 대한 95% 신뢰구간
### why df=n-1 을 따르는가? 
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

### 모분산 (3^2)를 알 때 : 모평균에 대한 95% 신뢰구간
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, csale=1) * 3 / np.sqrt(n)
