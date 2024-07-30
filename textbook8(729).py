#textbook 8, p.212
import pandas as pd
import seaborn as sns    
economics=pd.read_csv('data/economics.csv')
economics

economics.info()
sns.lineplot(data=economics, x="date",y="uempmed") # , cmap="Greens"
plt.show()
plt.clf()

import datetime
economics["date2"]=pd. to_datetime(economics["date"])

economics["month"]=economics["date2"].dt.month
economics["year"]=economics["date2"].dt.year
economics["dayofweek"]=economics["date2"].dt.dayofweek #0이 월요일
economics["day"]=economics["date2"].dt.day #() 안쓰는 이유 : 어트리뷰트 

economics["quarter"]=economics["date2"].dt.quarter
e_m=economics["month"].describe()


economics[["quarter","month"]]
economics.date2.dt.day_name()

economics['day']=economics['day'].astype(int)+3 # 시간 개념 고려 X
economics['date2']+pd.DateOffset(months=12) #윤년 등 시간 및 달력 개념 고려 

economics['date2'].dt.is_leap_year.value_counts() # 윤년 체크

sns.lineplot(data=economics, x="year", y="unemploy", errorbar=None)
plt.show()
plt.clf()

economics.unemploy.info()
#
sns.lineplot(data=economics, x="year", y="unemploy", size=1, color="red")
#
my_df=economics.groupby("year",as_index=False).agg(mon_mean=("unemploy", "mean"), mon_std=("unemploy","std"), mon_n=("unemploy", "count"))
my_df

my_df.loc[[48],"left_ci"]=my_df.mon_mean + 1.96*my_df['mon_std']/np.sqrt(4)
my_df["left_ci"]

my_df["right_ci"]=my_df.mon_mean - 1.96*my_df['mon_std']/np.sqrt(my_df["mon_n"])

mon_mean + 1.96*mon_std/sqrt(12)

import matplotlib.pyplot as plt
x=my_df['year']
y=my_df["mon_mean"]
plt.plot(x, y, color="black")
plt.scatter(x, my_df["left_ci"],s=3,color="red") #size 가 아니라 s=
plt.scatter(x, my_df["right_ci"],s=3, color="red")
plt.show()
plt.clf()
#
