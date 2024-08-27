#!pip install plotly
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np    

covid_100=pd.read_csv('data/df_covid19_100.csv')
covid_100

fig=go.Figure(
    data = [
        {"type": "scatter", 
         "mode": "markers",
         "x" : covid_100.loc[covid_100["iso_code"]=="KOR","date"],
         "y" : covid_100.loc[covid_100["iso_code"]=="KOR","new_cases"],
         "marker": {"color":"blue"}
         },
         {"type": "scatter", "mode": "lines",
         "x" : covid_100.loc[covid_100["iso_code"]=="KOR","date"],
         "y" : covid_100.loc[covid_100["iso_code"]=="KOR","new_cases"],
         "line" : {"color":'blue', 'dash' :'dash'}}
    ]
)
fig.show()

margins_P ={"t":50, "b": 25, "l": 25, "r":25}
fig=go.Figure(
    data= {
        "type": "scatter",
        "mode": "markers+lines",
        "x": covid_100.loc[covid_100["iso_code"]=="KOR","date"],
        "y": covid_100.loc[covid_100["iso_code"]=="KOR","new_cases"],
        "marker" : {"color":"#264E86"},
        "line" :{"color":"#5E88FC", "dash": "dash"}},
    layout={
        "title": "코로나 19 발생 현황",
        "xaxis":{"title":"날짜", "showgrid":False},
        "yaxis":{"title":"확진자수"},
        "margin": margins_P})
fig.show()


###

#프레임속성=
# 애니메이션 프레임 생성
frames = []
dates = covid_100.loc[covid_100["iso_code"] == "KOR", "date"].unique()

for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": covid_100.loc[(covid_100["iso_code"] == "KOR") & (covid_100["date"] <= date), "date"],
                "y": covid_100.loc[(covid_100["iso_code"] == "KOR") & (covid_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": covid_100.loc[(covid_100["iso_code"] == "KOR") & (covid_100["date"] <= date), "date"],
                "y": covid_100.loc[(covid_100["iso_code"] == "KOR") & (covid_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date)
    }
    frames.append(frame_data)

# 애니메이션을 위한 레이아웃 설정
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False},
    "yaxis": {"title": "확진자수"},
    "margin": margins_P,
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": covid_100.loc[covid_100["iso_code"] == "KOR", "date"],
            "y": covid_100.loc[covid_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": covid_100.loc[covid_100["iso_code"] == "KOR", "date"],
            "y": covid_100.loc[covid_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()

###
!pip install palmerpenguins
!pip install statsmodels

import pandas as pd
import numpy as np
import poltly.express as px
from palmerpenguins import load_penguins
import statsmodels.api as sm

penguins=load_penguins()
penguins.head()

fig=px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    trendline ="ols")


fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white")),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(title="펭귄 종",font=dict(color="white", size=12, family="맑은 고딕, Malgun Gothic, sns-serif")),
)

fig.show()


###
from sklearn.linear_model import LinearRegression    
model = LinearRegression()
penguins = penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]


model.fit(x,y)

linear_fit=model.predict(x)

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀 직선",
        line=dict(dash="dot", color="white"),
    )
)
fig.show()

#
penguins_dummies=pd.get_dummies(penguins,columns=["species"], drop_first=True) # drop_first = True 로 해도 잃는 정보 없음 

penguins_dummies.columns
penguins_dummies.iloc[:,-3:]


x=penguins_dummies[["bill_length_mm",'species_Chinstrap','species_Gentoo']]
y=penguins_dummies["bill_depth_mm"]
#model

model = LinearRegression()
model.fit(x,y)

model.coef_
model.intercept_
#y= model.coef_[0] * bill_lenngth + model.coef_[1] * species_Chinstrap  + model.coef_[2]  *  species_Gentoo

import matplotlib.pyplot as plt    
import seaborn as sns    

sns.scatterplot(x["bill_lenght_mm"],y,color = "black")
sns.scatterplot()
