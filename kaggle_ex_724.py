!pip install pandas
!pip install numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns    

<<<<<<< HEAD
train_row=pd.read_csv('data/house/train.csv')
train=train_row.copy()
test=pd.read_csv('data/house/test.csv')

# 결측치 둘다 없음
train["LandContour"].value_counts()
train["LandContour"].isnull().sum()

train["LandSlope"].value_counts()
train["LandSlope"].isnull().sum()

# 시각화화
plt.clf()
train_1=train[["LandContour", "SalePrice"]]
plt.scatter(data=train_1, x="LandContour", y= "SalePrice")
plt.show()
#
plt.clf()
train_2=train[["LandSlope", "SalePrice"]]



#
train1=train.groupby("LandContour")[["SalePrice"]].mean()
train2=train.groupby("LandSlope")[["SalePrice"]].mean()

plt.scatter(data=train1, x="LandContour", y= "SalePrice")
plt.scatter(data=train2, x="LandSlope", y= "SalePrice")
plt.show()

###########연진 코드

train=pd.read_csv("./data/houseprice/train.csv")
train=train[['Id','ExterCond','SalePrice']]
train.info()

# 결측치 확인
train['ExterCond'].isna().sum()

train = train.groupby('ExterCond', as_index=False) \
            .agg(mean_sale=('SalePrice', 'mean'))

train = train.sort_values('mean_sale', ascending=False)

sns.barplot(data=train, y='mean_sale', x='ExterCond', hue='ExterCond')
plt.show()
plt.clf()

'''

train_mean=train.groupby("YearBuilt")[["SalePrice"]].mean()
train[["Id","YearBuilt"]]
test=pd.read_csv('data/house/test.csv')
test=test[["Id", "YearBuilt"]]
new=pd.merge(test,train_mean,on="YearBuilt", how="left")
=======
train=pd.read_csv('data/house/train.csv')
train_mean=train.groupby("YearBuilt")[["SalePrice"]].mean()

train[["Id","YearBuilt"]]

test=pd.read_csv('data/house/test.csv')
test=test[["Id", "YearBuilt"]]

new=pd.merge(test,train_mean,on="YearBuilt", how="left")

>>>>>>> aa87cfc3d9d1d95a0b182a700d4d50954b0fc125
#결측치 전처리
cond1 =new["SalePrice"].isnull()
cond2 =(new["YearBuilt"]>=1900)
cond3 = (new["YearBuilt"]<1900)
new.loc[cond1&cond2, "SalePrice"] =  new.loc[cond2,"SalePrice"].mean()
new.loc[cond1&cond3, "SalePrice"] =  new.loc[cond3,"SalePrice"].mean()
<<<<<<< HEAD
new=new[["Id","SalePrice"]]
=======

new=new[["Id","SalePrice"]]

>>>>>>> aa87cfc3d9d1d95a0b182a700d4d50954b0fc125
new.info()
new["Id"]=new["Id"].astype(str)
new.sort_values(by="Id", ascending=False)
new.to_csv("sample_submission.csv", index=False)



<<<<<<< HEAD
# 수정정
#train["LandContour"]=np.where(train["LandContour"]=="Lvl","0",np.where(train["LandContour"]=="Bnk",1,np.where(train["LandContour"]=="HLS",2,3)))
# Lvl: 거의 플랫, Bnk: Banked - Quick and significant rise from street grade to building 
# HLS: Hillside - Significant slope from side to side  # Low : Depression
#train["LandSlope"]=np.where(train["LandSlope"]=="Gtl","0",np.where(train["LandSlope"]=="Mod",1,2))
#       Gtl	Gentle slope
#       Mod	Moderate Slope	
'''
#       Sev	Severe Slope

train =  pd.read_csv('data/train.csv')
#train = train[['Id', 'SaleType', 'ExterCond', 'GarageCars', 'LandContour', 'LandSlope', 'Neighborhood','SalePrice']]

# SaleType
SaleType_mean = train.groupby('SaleType', as_index = False) \
                     .agg(S_price_mean = ('SalePrice', 'mean'))
SaleType_mean = SaleType_mean.sort_values('S_price_mean', ascending = False)
sns.barplot(data = SaleType_mean, x = 'SaleType', y = 'S_price_mean', hue = 'SaleType')
plt.show()
plt.clf()

# ExterCond
ExterCond_mean = train.groupby('ExterCond', as_index=False) \
                      .agg(mean_sale=('SalePrice', 'mean'))

ExterCond_mean = ExterCond_mean.sort_values('mean_sale', ascending=False)

sns.barplot(data=ExterCond_mean, y='mean_sale', x='ExterCond', hue='ExterCond')
plt.show()
plt.clf()


# Ex > TA > Good> 각 개수가 몰려 있다.  




# GarageCars
GarageCars_mean = train.groupby('GarageCars', as_index = False) \
                             .agg(mean_price = ('SalePrice', 'mean')) \
                             .sort_values('mean_price', ascending = False)

sns.barplot(data = GarageCars_mean, x = 'GarageCars', y = 'mean_price', hue = 'GarageCars')
plt.show()
plt.clf()

# scatter 찍어보기기

LandContour_scatter1 = train[["LandContour", "SalePrice"]]
plt.scatter(data = LandContour_scatter1, x="LandContour", y= "SalePrice")
plt.show()
#
plt.clf()
LandContour_scatter2 = train[["LandSlope", "SalePrice"]]
plt.scatter(data = LandContour_scatter2, x="LandSlope", y= "SalePrice")
plt.show()


plt.clf()
LandContour_scatter3 = train[["SaleType", "SalePrice"]]
plt.scatter(data = LandContour_scatter3, x="SaleType", y= "SalePrice")
plt.show()



plt.clf()
LandContour_scatter5 = train[["Condition1", "SalePrice"]]
plt.scatter(data = LandContour_scatter5, x="Condition1", y= "SalePrice")
plt.show()


Neighborhood_mean = train.groupby('Neighborhood', as_index = False) \
                     .agg(N_price_mean = ('SalePrice', 'mean'))
####
plt.clf()
plt.grid()
sns.barplot(data = Neighborhood_mean, y = 'Neighborhood', x = 'N_price_mean', hue = 'Neighborhood')

LandContour_scatter4 = train[["Neighborhood", "SalePrice"]]
plt.scatter(data = LandContour_scatter4, y="Neighborhood", x= "SalePrice", s = 1, color = 'red')

plt.xlabel("price", fontsize=10)
plt.ylabel("n", fontsize=10)
plt.yticks(rotation=45,fontsize=8)
plt.show()


plt.clf()
sns.barplot(data = SaleType_mean, x = 'SaleType', y = 'S_price_mean', hue = 'SaleType')
plt.scatter(data = LandContour_scatter3, x="SaleType", y= "SalePrice", color = 'red')
plt.show()

plt.scatter(data=, x=train["OverallQual"],y= "SalePrice"])
=======


new["SalePrice"].fillna(new["SalePrice"].mean(), inplace=True)

new["SalePrice"].describe()


new.isnull().sum()

