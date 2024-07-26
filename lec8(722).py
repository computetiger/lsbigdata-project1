import pandas as pd
import matplotlib.pyplot as plt    
import seaborn as sns    
mpg=pd.read_csv('data/mpg.csv')
mpg

sns.scatterplot(data=mpg, x='displ',y='hwy')
plt.show()

sns.scatterplot(data=mpg,x="displ",y="hwy",hue="drv").set(xlim=[3,6], ylim=[10,30])
#plt.figure(figsize=(8,7))
plt.show()
#
plt.rcParams.update({'figure.dpi':"150", 'figure.figsize':[8,6]})
plt.show()

df_mpg=mpg.groupby("drv", as_index=False).agg(mean_hwy=('hwy','mean'))
mpg['drv'].unique()
mpg['drv'].nunique()

plt.clf()
sns.barplot(data=df_mpg, x="drv",y="mean_hwy",hue="drv")
plt.show()

df_mpg= df_mpg.sort_values("mean_hwy",ascending=False)
#208page
df_mpg=mpg.groupby("drv",as_index=False) \
          .agg(n=("drv","count"))    
sns.barplot(df_mpg,x="drv",y='n',hue="drv")
plt.show()
sns.countplot(mpg,x='drv',order=['4','f','r'])
plt.show()


###7.23
import numpy as np    
np.arange(33).sum()/33 #기댓값 구하는 방법
