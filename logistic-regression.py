import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir=os.path.dirname(cwd)
os.chdir(parent_dir)

admission_data=pd.read_csv("data/admission.csv")
print(admission_data.shape)
#합격을 한 사건: admit
# odds: 
p=admission_data.admit.sum()/ 400
p/(1-p)

admission_data["rank"].unique()
grouped_data=admission_data.groupby("rank", as_index="False").agg(p_admit=("admit","mean"))
grouped_data["odds"]=grouped_data['p_admit'] / (1 - grouped_data['p_admit'])

import seaborn as sns
sns.stripplot(x="rank",y="p_admit",jitter=0.3, data=grouped_data)
sns.regplot(data=grouped_data, x="rank", y="p_admit")                                    
grouped_data
grouped_data["log_odds"]=np.log(grouped_data["odds"])

import statsmodels.api as sm
admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')
model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()


odds=np.exp(-3.40 -0.058 * 0 + 0.002 * 450 + 0.075 * 3 -0.561 * 2)
pp=odds/ (1+odds)
odds

#문제

# Q. 1 
leukemia_df = pd.read_csv('data/leukemia_remission.txt', delimiter='\t')
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL +LI+ BLAST + TEMP", data=leukemia_df).fit()
print(model.summary())
# Q. 2 
# LLR p-value 가 0.1552 로 낮음. 유의함. 
# Q. 3
# 모든 변수의 p- value 가 0.05 이하 
# Q. 4
# 기존: leukemia_odds=np.exp(54.0042 + 23.0777*0.65 + 26.8239*0.45 -30.7709 *0.55 +2.8471*20000 -77.4924*38.5) 
leukemia_odds=np.exp(64.2581 + 30.8301*0.65 + 24.6863*0.45 -24.9745 *0.55 +4.3605*1.2 -0.0115*1.1 -100.1734*0.9) 
leukemia_odds #0.03817459641135519
leukemia_p =leukemia_odds / (leukemia_odds+ 1)
leukemia_p  # 0.03677088280074742
# Q. 6
np.exp(100.1734) 
# Q. 7
from scipy.stats import norm
a=30.8301 - norm.ppf(0.995)* 52.135
b=30.8301 + norm.ppf(0.995)* 52.135
print("신뢰구간:",a,"~",b)
# Q. 8
from sklearn.metrics import confusion_matrix
pred_y = model.predict(leukemia_df)
pred_2 = [1 if i > 0.5 else 0 for i in pred_y]

conf_mat = confusion_matrix(leukemia_df['REMISS'], pred_2)
conf_mat

# Q. 9 Accuracy
accuacy=(15 + 5)/ (15 + 3 + 4 + 5)
print("Accuracy:",round(accuacy,3))
# Q. 10 F1 score
recall=15/19
precision=15/18
f1=2*(recall * precision)/ (recall + precision)
print("f1-score:",round(f1,3))