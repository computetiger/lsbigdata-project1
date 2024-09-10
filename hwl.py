
#문제
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Q. 1 
leukemia_df = pd.read_csv('data/leukemia_remission.txt', delimiter='\t')
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL +LI+ BLAST + TEMP", data=leukemia_df).fit()
print(model.summary())

# Q. 2 
print(" LLR p-value: 0.046 (p-vlaue : 0.05)")
# Q. 3
# 모든 변수의 p- value 가 0.05 이하 
print("LI: 0.101, TEMP: 0.198")
# Q. 4
leukemia_odds=np.exp(64.2581 + 30.8301*0.65 + 24.6863*0.45 -24.9745 *0.55 +4.3605*1.2 -0.0115*1.1 -100.1734*0.9) 
leukemia_odds #0.03817459641135519
leukemia_p =leukemia_odds / (leukemia_odds+ 1)
leukemia_p  # 0.03677088280074742
# Q. 6
np.exp(100.1734) 
# Q. 7
cell_p=leukemia_df.CELL.mean()
cell_odds=cell_p/(1-cell_p)
a=cell_odds - 


from scipy.stats import norm
a=cell_odds - norm.ppf(0.995)* 52.135
b=cell_odds + norm.ppf(0.995)* 52.135
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