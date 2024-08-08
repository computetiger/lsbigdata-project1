import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
# y= a* x + b
plt.clf()
#
a = 28
b = 92
x=np.linspace(0, 8, 100)
y= a* x + b
plt.plot(x, y, color ="black")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
#
train[["BedroomAbvGr"]].value_counts()
# 가장 높은 값인 3, 2, 4 순서만 고려하기. 
train=pd.read_csv("data/house/train.csv")
train1=train[["BedroomAbvGr","SalePrice"]]
train1["SalePrice"]=train1["SalePrice"]/1000
train1["BedroomAbvGr"].value_counts()
plt.scatter(x=train1["BedroomAbvGr"], y=train1["SalePrice"], s=3, alpha=0.5, color="black")
plt.show()
# 평균값 추출 - red
train_mean=train.groupby("BedroomAbvGr")[["SalePrice"]].mean().astype(int)
train_mean["SalePrice"]=train_mean["SalePrice"]/1000
#plt.plot(train_mean["BedroomAbvGr"], train_mean["SalePrice"], color="red")
plt.plot(train_mean, color="red")
plt.show()
# 평균값 x = 2 158.2
# 중앙값 x =2 137.2
# 둘의 평균 = 147.7
# 평균값 x = 3 181
# 중앙값 x =3 170
# 둘의 평균 = 175.5
# 평균값 (x=2, x=3)의 방정식: 
# 즉, (2,147.7) , (3, 175.5) 를 지나는 직선이 가장 회귀직선에 가까울 것이다. 

# y = 23x + 113 (평균)                  81868078
# y = 33x + 72 (중앙값)                 80745406
# y = 28x + 92 (평균-중앙값의 평균)     80705782
# y = 27.8x + 92.1 (평균-중앙값의 평균) 80618782
# y = 29.6x + 87.9 (편균- 중앙값의 평균, 단 2,3,4) 80930348
# 즉, 평균- 중앙값의 평균이 가장 최적화 선. 
# 중앙값 추출 - green
train_median=train.groupby("BedroomAbvGr")[["SalePrice"]].median().astype(int)
train_median["SalePrice"]=train_median["SalePrice"]/1000
plt.plot(train_median, color="green")
plt.show()
#
real_train=(train_median["SalePrice"]+train_mean["SalePrice"]) / 2

#
a=27.8
b=92.1
test =  pd.read_csv('data/house/test.csv')
test_df = test[["Id", "BedroomAbvGr"]]
test_df['SalePrice'] = (a * test_df.loc[:, ['BedroomAbvGr']] + b) * 1000
test_df = test_df[["Id", "SalePrice"]]

test_df.to_csv('sample_submission5.csv', index = False)


#직선 성능 평가 
a=29.6
b=87.9

# y_hat 어떻게 구할까? 
y_hat=(a * train["BedroomAbvGr"] + b) *1000
# y는 어디에 있는가? 
y=train["SalePrice"]
#
sum(np.abs(y - y_hat))


## 0802
