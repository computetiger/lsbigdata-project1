import pandas as pd
import numpy as np    

exam=pd.read_csv("data/exam.csv")
#판다스 데이터프레임으로 만듦. 
exam.head()

exam.head()
exam.tail()
exam.shape
exam.info()
exam.describe()

exam.info

type(exam)

exam2=exam.copy()
exam.rename(columns={"nclass":"class"}, inplace=True)
exam2['total']=exam2['math']+exam2['english']+exam2['science']
exam2.head()

#exam2['test']=np.where(exam2["total"]>200,"pass","fail")

exam2['test']="pass" if (exam2["total"]>200).astype(bool) else "fail"

exam2['test']=exam2['total'].apply(lambda x: "pass" if x>200 else "fail")
exam2



###
def myfun(a):
    if a ['total']>=200:
        return "pass"
    else:
        return "fail"
exam2['test']=exam2.apply(myfun, axis=1)
###
import matplotlib.pyplot as plt
count_test=exam2.test.value_counts()
count_test.plot.bar(rot=0)
plt.ylabel("students")
plt.show()
plt.clf()

exam2['test2']=np.where(exam2['total']>=200,"A", np.where(exam2['total']>=100,"B","C"))
exam2['test2'].isin(['A','C'])
