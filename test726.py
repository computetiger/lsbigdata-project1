from scipy.stats import uniform
import numpy as np    
import matplotlib.pyplot as plt    
uniform.rvs(loc=2, scale=4, size=1)



x=np.linspace(0,8,100)
uniform.pdf(x, loc=0, scale=1)
uniform.cdf(x, loc=0, scale=1)
uniform.ppf(q, loc=0, scale=1)
uniform.rvs(loc=0, scale=1, size=None, random_stat)


k=np.linspace(0,8,100)
y=uniform.pdf(k, loc=2, scale=4) #loc: 위치 하한선, 분포가 2에서 시작. , 
#scale 분포의 길이, 즉 여기서는 4니까 하한 2, 상한 6이라는 의미. 
plt.plot(k,y, color="red")
plt.show()
uniform.cdf(3.25,loc=2,scale=4) # why loc와 scale 값이 이렇지? uniform = 균일분포 
uniform.cdf(8.39,loc=2,scale=4)-uniform.cdf(5,loc=2,scale=4)
#95% 신뢰구간
4-norm.ppf(0.025,loc=4, scale=np.sqrt(1.333333/20))
4-norm.ppf(0.975,loc=4, scale=np.sqrt(1.333333/20))
# 99% 신뢰구간
4-norm.ppf(0.005,loc=4, scale=np.sqrt(1.333333/20))
4-norm.ppf(0.995,loc=4, scale=np.sqrt(1.333333/20))
# 표본 20개뽑고 표본평균 계산 
plt.clf()
x=uniform.rvs(loc=2,scale=4,size=20*1000, random_state=42)
x=x.reshape(-1,20)
x.shape
x.mean(axis=1).shape
blue_x=x.mean(axis=1)
blue_x

import seaborn as sns
sns.histplot(blue_x, stat="density")
plt.show()
# X bar ~ N(mu, sigma^/n)
# X bar ~ N(4, 1.33333/20)
uniform.var(loc=2, scale=4)
uniform.expect(loc=2,scale=4)
##
pdf_values=norm.pdf(x_values,loc=4,scale=np.sqrt(1.333))
# 신뢰구간
from scipy.stats import *
x_values=np.linspace(2,6,100)
pdf_values=norm.pdf(x_values,loc=4,scale=np.sqrt(1.333333/20))
plt.plot(x_values, pdf_values, color = "red", linewidth=2)


#######

plt.axvline(x=4, color="green",linestyle="--", linewidth=3)
plt.scatter(blue_x,0.002,color="blue", zorder=10, s=10)
plt.show()
##### copy()

var = uniform.var(2, 4)
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(var/20))
plt.plot(x_values, pdf_values, color = 'red', linewidth = 2)


# 표본 평균 포인트 찍기
blue_x = uniform.rvs(loc=2, scale = 4, size = 20).mean()
plt.scatter(blue_x, 0.002, color ='green', zorder=10, s=10)

#기대값 표현
plt.axvline(x = 4, color = 'blue', linestyle = '-', linewidth = 2)

plt.show()
plt.clf()
# 95% 커버, a.b 표준편차 기준 몇 배? 
var = uniform.var(2, 4)
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(var/20))
plt.plot(x_values, pdf_values, color = 'red', linewidth = 2)

# 2.58 추정
norm.ppf(0.995, loc=0, scale=1)
# 1.96 추정
norm.ppf(0.975, loc=0, scale=1)
