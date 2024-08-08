# import 
import numpy as np    
import pandas as pd
# file read
tab3=pd.read_csv('data/tab3.csv')
tab3

tab1=tab3[["id","score"]]
tab1["id"]=np.arange(1,13)
tab1

tab2=tab1.assign(gender=["female"]*7 + ["male"]*5)
tab2


# 1
## 1표본 t 검정 (그룹 1개)
## 귀무가설 vs 대립가설
## H0 : mu =10   "vs"   Ha: mu != 10
## 유의수준 5% 로 설정 

from scipy.stats import ttest_1samp  
 # tab1 은 No (id 도 들어가서)
result = ttest_1samp (tab1["score"], popmean=10, alternative = "two-sided")
print("p-value:", result[1])

'''
alternative : 대립가설의 의미 
two-sided, less, greater
첫번째 작성한 것이 기준이 됨. 
첫번쨰(왼쪽) 쓴 게 보다 작으면 less, 크면 greater 
'''
# result 안에 있는 값
t_value=result[0] # t 검정통계량 
p_value=result[1] # 유의확률  p- value
tab1["score"].mean() # 표본평균 = 11.53

# 메서드로 호출 가능
result.pvalue
result.statistic
result.df

'''
유의확률 0.0648 이 유의수준 0.05 보다 큼. 귀무가설 기각하지 못한다. 
귀무가설(mu = 10)이 참일 때, 11.53이 관찰될 확률이 6.48% 이므로, 유의수준 0.05보다 크기에 
귀무가설을 거짓이라 판단하기 힘들다. 
'''

# 신뢰구간 0.95
ci=result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]

#
## 2표본 t 검정 (그룹 2)
## 분산 같은경우: 독립 2표본 t검정
## 분산 다를경우: 웰치스 t 검정
## 귀무가설 vs 대립가설
## (H0 :mu_male = mu_female ) vs  (Ha: mu_male  > mu_female)
tab2_m=tab2[tab2["gender"]=="male"]["score"]
tab2_f=tab2[tab2["gender"]=="female"]["score"]
result2=ttest_ind(tab2_m,tab2_f,alternative="greater", equal_var=True)
#
result2.pvalue
result2.statistic
result2.df
ci2=result2.confidence_interval(0.95)



# 3
## 대응표본 t 검정 ( 짝지을 수 있는 표본)
# 귀무가설  vs 대립가설 
# ( H0: mu_before = mu_after ) vs ( Ha: mu_after > mu_ before )
# (H0: mu_diff = 0)  vs  (Ha: mu_diff >0)
# mu_diff = mu_after - mu_before
# 유의수준 5% 로 설정 

# data - pivot 으로 변환, mu_diff  추가 
tab3_data=tab3.pivot_table(index="id", columns="group", values="score").reset_index()
tab3_data["score_diff"]=tab3_data["after"]-tab3_data["before"]
tab3_data.melt(id_vars="id", value_vars=["before","after"], var_name="group", value_name="score")

# t- test 1 표본
result3 = ttest_1samp (tab3_data["score_diff"], popmean=10, alternative = "greater")
result3.pvalue



# 연습 2
#!pip install seaborn
import seaborn as sns   
tips=sns.load_dataset("tips")
#
tips.pivot_table(index="day",columns="time", values="tip",aggfunc="sum", margins=True)

