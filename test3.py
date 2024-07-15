#교재 63p
import seaborn as sns
import matplotlib.pyplot as plt

var=['a','a','b','c']

sns.countplot(x=var)
plt.show()

import os
print(os.getcwd())

import pandas as pd


df=sns.load_dataset('titanic')
df
plt.clf()
sns.countplot(data=df, y='pclass', hue='pclass',orient="v")
#sns.countplot(data=df,x='pclass', hue='alive',color="00000")
plt.show()

#해당 함수를 모를때는 제일 앞에 물음표를 붙여주면 된다. 
?sns.countplot

#!pip install scikit-learn

import sklearn.metrics
sklearn.metrics.accuracy_score() #이것과
from sklearn import metrics #이것은 같다. 
import sklearn.metrics as met




