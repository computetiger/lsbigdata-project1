#1. 지도시각화 ( 교재 11)
import json 
geo = json.load(open("data/SIG.geojson", encoding="UTF-8"))

# 행정 구역 코드 출력
geo["features"][0]["properties"]
# 위도, 경도 좌표 출력
geo["features"][0]["geometry"]

import pandas as pd
df_pop=pd.read_csv('data/Population_SIG.csv')
df_pop.head()
df_pop["code"] = df_pop["code"].astype(str)
#
!pip install folium
import folium 
folium.Map(location = [35.95, 127.7],
           zoom_start =8 )

map_sig = folium.Map(location=[35.95, 127.7],
                    zoom_start = 8,
                    tiles = "cartodbpositron")
map_sig


#
folium.Choropleth(
    geo_data = geo,
    data = df_pop,
    columns =("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
map_sig

# 계급 구간 정하기 
bins = list(df_pop["pop"].quantile([0,0.2,0.4,0.6,0.8,1]))
bins

# 배경 지도 만들기
map_sig = folium.Map(location = [35.95, 127.7], #지도 중심 좌표
                     zoom_start = 8,    # 확대 단계
                     tiles ="cartodbpositron") # 지도 종류

# 단계 구분도 만들기
folium.Choropleth(
    geo_data = geo,
    data = df_pop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD",
    fill_color = "YIGnBu",
    fill_opacity = 1,
    line_opacity = 0.5,
    bins = bins).add_to(map_sig)
map_sig






# 2. 교재 11-2, 서울시 지도 시각화 
import json
import matplotlib.pyplot as plt    
import numpy as np    

geo_seoul=json.load(open('data/SIG_Seoul.geojson', encoding = "UTF-8"))
type(geo_seoul)
#
len(coordinate_list)           # 1
len(coordinate_list[0])        # 1
len(coordinate_list[0][0])     # 2332
coordinate_list[0][0][0]
#

coordinate_list=geo_seoul["features"][2]["geometry"]["coordinates"]
# np
x=np.array(coordinate_list[0][0])[:,0]
y=np.array(coordinate_list[0][0])[:,1]

plt.clf()
plt.plot(x, y, c="blue")
plt.show()

#
def draw_seoul(x):
    
    name = geo_seoul["features"][x]["properties"]["SIG_KOR_NM"]
    
    coor_list=geo_seoul["features"][x]["geometry"]["coordinates"]
    x=np.array(coor_list[0][0])[:,0]
    y=np.array(coor_list[0][0])[:,1]
    
    plt.rcParams.update({"font.family":"Malgun Gothic"})
    plt.plot(x,y)
    plt.title(name)
    plt.show()
    plt.clf()


draw_seoul(5)

import pandas as pd
# 팀 숙제
df=pd.DataFrame({})
df["lo"]=np.array(coor_list[0][0])[:,0]
#
for i in range
    
    #name = geo_seoul["features"][x]["properties"]["SIG_KOR_NM"]
    
    coor_list=geo_seoul["features"][x]["geometry"]["coordinates"]
    x=np.array(coor_list[0][0])[:,0]
    y=np.array(coor_list[0][0])[:,1]
#### ctrl + c, v
def df_gu(x):
    import numpy as np
    import pandas as pd
    coordinate_list = geo_seoul["features"][x]["geometry"]["coordinates"][0][0]
    coordinate_array = np.array(coordinate_list)
    df = pd.DataFrame({})
    df["gu_name"] = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"]]*len(coordinate_array)
    df["x"] = coordinate_array[:,0]
    df["y"] = coordinate_array[:,1]
    return df
df_gu(0)
result = pd.DataFrame({})
for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_gu(x)])
    df = df_gu(x)
    plt.plot(df["x"],df["y"])
plt.show()
result = result.reset_index(drop=True)
result

# 1 
plt.plot(result['x'],result['y'])
#sns.lineplot(data = result,x = 'x', y = 'y', hue= "gu_name")
plt.legend(fontsize = 2)
plt.show()
plt.clf()


# 2
for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_gu(x)])
sns.scatterplot(data = result,x = 'x', y = 'y', hue= "gu_name")
plt.legend(fontsize = 2)
plt.show()
plt.clf()
px.scatter(data_frame = result, x = 'x', y = 'y', color = 'gu_name')
plt.show()
plt.clf()

### 교재 (수업)
import numpy as np
import matplotlib.pyplot as plt
import json

df_pop=pd.read_csv('data/Population_SIG.csv')
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"]=df_seoulpop["code"].astype("str")
df_seoulpop["code"].info()

#!pip install folium
import folium

my_map = folium.Map(location = [37.551, 126.973], zoom_start =9, tiles="cartodbpositron")
my_map.save("map_seoul.html")
# 코로플릿
folium.Choropleth(geo_data=geo_seoul, data=df_seoulpop, fill_color="viridis",bins=bins, columns=("code", "pop"), key_on = "feature.properties.SIG_CD").add_to(map_sig)
folium.Marker([37.583744, 126.983800], popup="강남구").add_to(map_sig)

map_sig.save("map_seoul.html")

bins = df_seoulpop["pop"].quantile([0,0.2,0.4,0.6,0.8,1])







