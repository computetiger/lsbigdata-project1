import pandas as pd
import numpy as np    
from scipy.stats import *
import matplotlib.pyplot as plt    

# 2. 검정을 위한 가설을 명확하게 설명하라. 
# Null-H: 현대자동차의 신형 모델은 에너지 소비효율이 16.0 이상이다. 
#         x>=16
# Alternative-H : 현대자동차의 신형 모델은 에너지 소비효율이 16.0 미만이다.
#         x< 16

# 3. 검정통계량 계산하시오.

car=np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, \
    16.93, 14.118, 14.927, 15.382, 16.709, 16.804])

mean=np.mean(car)
n=len(car)
sigma=np.std(car,ddof=1)/np.sqrt(n)

t_value = (np.mean(car)-16) / (np.std(car, ddof=1) / np.sqrt(len(car)))
print("검정통계량:", t_value)


# 4. p-value 를 구하시오.  # 유의수준 1 % = 0.01
p_value=t.cdf(t_value,n-1)
print("p-vlaue:", p_value)
print("유의수준 0.01 보다 p-value가 크기 때문에 귀무가설을 기각할 수 없다.")

# 4-1. p-value 정규분포 
x=np.linspace(14,17,400)
y=norm.pdf(x,loc=mean,scale=sigma)
plt.plot(x,y,color="black")
plt.axvline(16, linestyle="--",linewidth="2",color="green")
plt.show()

# 6. 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간을 구해보세요.

car_max= mean + t.ppf(0.975, n-1)  * sigma
car_min= mean + t.ppf(0.025, n-1) * sigma
print("신뢰구간:",car_min,"~",car_max)

# 6-1. 95% 신뢰구간 그래프 
plt.plot(x,y,color="black")
plt.axvline(car_min, linestyle="-",linewidth="2",color="red")
plt.axvline(car_max, linestyle="-",linewidth="2",color="red")
plt.show()
