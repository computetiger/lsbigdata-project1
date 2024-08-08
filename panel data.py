import pandas as pd
import numpy as np
import seaborn as sns    
import matplotlib.pyplot as plt    
!pip install pyreadstat
raw_welfare=pd.read_spss('data/Koweps_hpwc14_2019_beta2.sav')
welfare=raw_welfare.copy()
# welfare 의 특징
welfare.shape
welfare.info()
welfare.describe()
#
welfare=welfare.rename(
    columns={"h14_g3": "sex",
             "h14_g4": "birth",
             "h14_g10": "marriage_type",
             "h14_g11": "religion",
             "p1402_8aq1": "income",
             "h14_eco9" : "code_job",
             "h14_reg7" : "code_region"})
             
welfare=welfare[["sex","birth","marriage_type","religion","income","code_job","code_region"]]

# 성별 전처리
welfare["sex"].value_counts()
welfare["sex"].isna().sum()
welfare["sex"]=np.where(welfare["sex"]==1, "male","female")
# 성별 빈도 막대 그래프
sns.countplot(data= welfare, x= "sex")
plt.show()

#월급 시각화
plt.clf()
welfare["income"].describe()
sns.histplot(data=welfare, x="income")
plt.show()
'''
# 월급 전처리 & 월급 평균 
welfare["income"].isna().sum()
sex_income=welfare.dropna(subset="income").groupby("sex",as_index=False).agg(mean_income=("income", "mean")) 
sex_income
# 성별 월급 평균 그래프 
plt.clf()
sns.barplot(data=sex_income, x="sex", y="mean_income")
plt.show()
'''
# 나이 변수 시각화 
plt.clf()
welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
# 결측치 확인
welfare["birth"].isna().sum()
# 나이 칼럼 만들기 & 시각화
plt.clf()
welfare=welfare.assign(age=2019-welfare["birth"]+1)
welfare.describe()
sns.histplot(data=welfare, x="age")
plt.show()
#나이별 월급 평균표 만들기
plt.clf()
age_income=welfare.dropna(subset="income").groupby("age").agg(mean_income=("income","mean"))
sns.lineplot(data=age_income, x="age", y="mean_income")
plt.show()
# 나이별 income 칼럼 na개수 세기
my_df=welfare.assign(income_na=welfare["income"].isna()) \
                .groupby("age", as_index= False) \
                .agg(n=("income_na", "count"))

# 9-6
sns.barplot(data=my_df, x="age", y="n")
plt.show()

# 240-241쪽, 연령대에 따른 월급 차이
welfare["age"].head()
#welfare["income"].isna().sum()
plt.clf()
welfare=welfare.assign(age_group=np.where(welfare["age"]< 30, "young", np.where(welfare["age"]<=59, "middle", "old")))
welfare["age_group"].value_counts()

sns.countplot(data=welfare, x= "age_group")
plt.show()
# 연령대별 월급 평균표
plt.clf()
age_group_income=welfare.dropna(subset="income").groupby("age_group", as_index=False).agg(mean_income=("income", "mean"))
sns.barplot(data=age_group_income, x="age_group",y="mean_income" )
plt.show()
# 문제 : 10단위로 나이 나누어 그룹화 (10-19: 10대)
# 시각화
def age_f(x):
    age_list=[]
    for i in x:
        y=i//10
        z=f"{y*10}s"
        age_list.append(z)
    return age_list

plt.clf()
welfare["age_range"]=age_f(welfare["age"])
welfare["age_range"].isna().sum()
range_income=welfare.dropna(subset="income").groupby("age_range")[["income"]].mean()
range_income.rename(columns={"income":"a_g_income"}, inplace=True)
sns.barplot(data=range_income, x="age_range",y="a_g_income",palette="Greens")
plt.show()
range_income.dtypes

# 10단위로 나눈 그룹에 성별 추가, 시각화
plt.clf()
range_incomes=welfare.dropna(subset="income").groupby(["age_range","sex"],as_index=False)["income"].mean()
sns.barplot(data=range_incomes, x="age_range",y="income",hue="sex", palette="Set2")
plt.show()
plt.clf()
range_incomes

# 연령대별, 성별 상위 4% 수입 찾기 # groupby () 안에 as_index=False
range_incomes_q=welfare.dropna(subset="income").groupby(["age_range","sex"] ,as_index=False)[["income","code_job"]].agg(["std","mean"])
sns.barplot(data=range_incomes_q, x="age_range",y="income",hue="sex", palette="Set3")
plt.show()
plt.clf()
#
# welfare.info()
# range_incomes_q["income"].iloc[2,1]

# 9-6
# !pip install openpyxl
welfare["code_job"].value_counts()
list_job=pd.read_excel("./data/koweps/Koweps_Codebook_2019.xlsx", sheet_name="직종코드")
list_job.head()
welfare=welfare.merge(list_job, how="left", on="code_job")
welfare.dropna(subset=["job","income"])[["income","job"]]

# 직업별 월급 평균 표 만들기 & query를 사용해 성별 추출
job_income=welfare.dropna(subset=["job","income"]).query("sex=='male'").groupby("job")[['income']].mean()
#직업별 월급 평균 top 10
job_10=job_income.sort_values("income",ascending=False).head(10)
# font 설정
plt.rcParams.update({'font.family': "Malgun Gothic"})
sns.barplot(data=job_10, y="job", x="income", hue= "job")
plt.show()
plt.clf()
welfare["marriage_type"]
#종교 유무에 따른 이혼율표
rel_div = welfare.query("marriage_type != 5").groupby("religion", as_index=False)["marriage_type"].value_counts(normalize=True)
# 이혼만 추리기 # 백분율로 환산
rel_div.query("marriage_type==3.0").assign(proportion = rel_div["proportion"]*100).round(1)

#
rel_div.query("marriage_type==1.0 | marriage_type==4.0").assign(proportion = rel_div["proportion"]*100).round(1)
