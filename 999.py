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
import pandas as pd
import numpy as np
import poltly.express as px
from palmerpenguins import load_penguins()