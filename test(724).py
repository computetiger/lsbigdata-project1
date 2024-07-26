!pip install scipy
import numpy as np    
import pandas as pd
from scipy.stats import bernoulli

#P(X=1)
bernoulli.pmf(1,0.3)
#P(X=0)
bernoulli.pmf(0,0.3)
# X ~ B(n,p)    -(Binary)
from scipy.stats import binom
b=[]
for i in range(31):
    a=binom.pmf(i,n=30,p=0.3)
    b.append(a)
b
a=[binom.pmf(i,30,0.3) for i in range(31)]
#
import math
math.factorial(54)/(math.factorial(26)*math.factorial(28))
math.comb(54,26) # 54C26

n=np.cumprod(np.arange(1,55))[-1]
k=np.cumprod(np.arange(1,27))[-1]
k_1=np.cumprod(np.arange(1,29))[-1]
n/(k*k_1)

k=math.log(54)-math.log(26)-math.log(28)
10^k
a=sum(np.log(np.arange(1,55)))-sum(np.log(np.arange(1,27)))-sum(np.log(np.arange(1,29)))
np.exp(35.168)

math.comb(2,0)*0.3*(1-0.3)**2
#######
math.comb(2,0) * (0.3 ** 0) * (0.7 ** 2)  #0.49
 
math.comb(2,1) * (0.3 ** 1) * (0.7 ** 1)  #0.42

math.comb(2,2) * (0.3 ** 2) * (0.7 ** 0)  #0.09
#  n C k * p ^ k  *(1 - p) ^(n-k)
binom.pmf(4,n=10,p=0.36)
#
#P(X<=4) = ?
binom.pmf(np.arange(5),n=10,p=0.36).sum()
# P(2<X<=8) = ?
binom.pmf(np.arange(3,9),n=10,p=0.36).sum()
#
binom.pmf(np.arange(4),n=30,p=0.02).sum()+ binom.pmf(np.arange(25,31),n=30,p=0.02).sum()
1-binom.pmf(np.arange(4,25),n=30,p=0.02).sum()
# rvs: random variates sample. 표본 추출 함수 
binom.rvs(n=2,p=0.3, size=1) # size: 한 번에 여러개를 뽑고 싶기 때문에. 
#X ~B(30,0.26)
# 표본 30개 추출
binom.rvs(n=30,p=0.26,size=30).mean()
#베르누이 확률 변수/ 이항분포의 평균은 np// 분산( np(1-p) )
# ㅁㅊ 다까먹음 
!pip install seaborn
import seaborn as sns    
sns.barplot(binom.rvs(n=30,p=0.26,size=30))
#
sns.barplot(binom.pmf(np.arange(31),n=30,p=0.26))
plt.show()
#
binom.pmf(np.arange(31),n=30,p=0.26)
df=pd.DataFrame({"values":np.arange(31), "probab":binom.pmf(np.arange(31),n=30,p=0.26)})
sns.barplot(df,x="values",y="probab")
#
plt.clf()
!pip install matplotlib
import matplotlib.pyplot as plt
# cdf: cumulative dist. function 
# 누적확률분포 함수
# F_X(x) = P(X<=x)
# F_4(x) = P(X<=x)
# P(4<x<=18)= ? 
binom.cdf(18,n=30,p=0.26)-binom.cdf(4,n=30,p=0.26)
binom.cdf(19,n=30,p=0.26)-binom.cdf(13,n=30,p=0.26)
x=np.arange(31)
prob_x=binom.pmf(x,n=30,p=0.26)
sns.barplot(prob_x,color="blue")
---
plt.scatter(a,np.repeat(0.002,10),color="red",zorder=100, s=5)

binom.cdf(20,n=30,p=0.26)
a=binom.rvs(n=30,p=0.26, size=10) # size: 한 번에 여러개를 뽑고 싶다는 옵션. 
a
plt.show()
#X ~B(30,0.26)
---

#기댓값 표현 
plt.axvline(x=7.8, color='green', linestyle="--",linewidth=2)
plt.show()
plt.clf()
---
x_1=binom.rvs(n=30, p=0.26, size=10)

binom.ppf(0.7, n=30, p=0.26)
binom.cdf(0.7,n=30, p=0.26)

1/math.sqrt(2*math.pi)*math.exp((-1/2)*(x)**2)

1/np.sqrt(2*math.pi)

import pandas as pd
from scipy.stats import norm # norm : 정규분포 
norm.pdf(0, loc=0,scale=1)
loc: mu , scale: sigma 
norm.pdf(5, loc=3,scale=4)
#정규분포 
plt.clf()
k=np.linspace(-5,5,300)
y=norm.pdf(k,loc=0,scale=1)
y1=norm.pdf(k, loc=0,scale=2)
y4=norm.pdf(k, loc=0,scale=0.5)

plt.plot(k,y, color="black")
plt.plot(k,y1, color="red")
plt.plot(k,y4, color="green")

plt.show()

#>> 분포의 모양을 결정하는 것은 평균과 표준편차니깐 linspace 조절은 틀림
k2=np.linspace(-5,5,100)
y2=norm.pdf(k2,loc=0,scale=1)
plt.scatter(k2,y2, color="green")
plt.show()

norm.cdf(0.54,loc=0,scale=1)-norm.cdf(-2,loc=0,scale=1)
1-norm.cdf(3,loc=0,scale=1)+norm.cdf(1,loc=0,scale=1)

norm.cdf(5,loc=3,scale=5)-0.5
#X ~ N(3,5^2)
norm.cdf(5,3,5)-norm.cdf(3,3,5) # 0.155
#위 확률변수에서 표본 100개를 뽑기. 
x=norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3)& (x<5))/1000 # 0.156
# 표준정규분포를 따르는 정규분포에서 표본을 1000개 뽑아 0보다 작은 비율은 표준정규분포에서 P(X<=0)의 
# 면적과 같다. 
norm.cdf(0,0,1)
y=norm.rvs(loc=0, scale=1, size=1000)
sum(y<0)/1000
np.mean(y<0)
-
######
plt.clf()
x=norm.rvs(loc=3, scale=2,size=1000)
-
xmin, xmax =(x.min(),x.max())
x_values=np.linspace(xmin,xmax, 100)
pdf_values=norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values,pdf_values, color='red',linewidth="2")
sns.histplot(x, stat="density")
plt.show()
