import numpy as np
import matplotlib.pyplot as plt
# example 1
data=np.random.rand(10)
plt.clf()
plt.hist(data, bins=5,alpha=0.7, color="red") # bins: 데이터의 분포를 나누는 구간. 
plt.title("histogram of numpy vector")
plt.xlabel('value')
plt.ylabel("Frequency")
plt.grid(True) # grid: 격자 
plt.show()

# example 2
data=np.random.rand(10)
############################# 왜안되니? 
def x(a):
    np.random.seed(2024)
    b=np.random.rand(a).mean()
    return b

x(5)

bins =np.linspace(0,1, 21)
>>>>>c=np.arange(x(1000))*10000
plt.hist(a, bins=bins,alpha=0.7, color="red")
plt.grid()
plt.show()
plt.clf()
### 망해버림 . ㅜㅜ


# 정규분포 그리기
plt.clf()
a=np.random.rand(10000,5).mean(axis=1)
# bins =np.linspace(0,1, 21)
plt.hist(a, alpha=0.7, color="red")
plt.grid()
plt.show()

#ex
import pandas as pd
df=pd.DataFrame({'sex':[3,4,67,8,5,7,9,6,0,4,2,5,78,9,4,2,1],
                 'score': [5,4,3,4,0]})
