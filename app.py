
import streamlit as st
from PIL import Image
import pandas as pd

st.write('Hello, *World!* :sunglasses:') # 해당 내용을 수정해서 사이트를 자유롭게 꾸밀 수 있다.

st.title('불량 원인 파악 및 공정 최적화')
st.header('5th Project: 강남규 김연예진 김재희 박수빈 이재준 최지원 ')
st.subheader('LS Bigdata School 3rd')
# 탭 생성 : 첫번째 탭의 이름은 Tab A 로, Tab B로 표시합니다.
tab1, tab2= st.tabs(['Tab A' , 'Tab B'])

with tab1:
  #tab A 를 누르면 표시될 내용
  st.write('hello_new_version')
  import matplotlib.pyplot as plt

  # 폰트 설정
  plt.rcParams['font.family'] = 'Malgun Gothic'
  plt.rcParams['axes.unicode_minus'] = False

  # 데이터 예시 (data['passorfail']에 해당하는 데이터프레임이 이미 있다고 가정)
  # 실제 사용 시에는 data를 Streamlit 앱 내에서 정의하거나 파일로부터 읽어와야 합니다.
  data = pd.DataFrame({'passorfail': [0]*87998 + [1]*4016})

  # passorfail의 값 개수 계산
  passorfail_counts = data['passorfail'].value_counts()

  # 색상 설정
  pastel_colors = ['#ff9999', '#99ccff']

  # Streamlit 앱 설정
  st.title("Pass or Fail Count Visualization")

  # Matplotlib 그래프 설정
  fig, ax = plt.subplots(figsize=(8, 6))
  bars = ax.bar(passorfail_counts.index, passorfail_counts.values, tick_label=['정상(0)', '불량(1)'], color=pastel_colors)

  # 막대 위에 값 표시
  for bar in bars:
      yval = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')

  # 제목과 y축 레이블 설정
  ax.set_title("Pass or Fail 개수 시각화")
  ax.set_ylabel("개수(count)")

  # Streamlit에 그래프 표시
  st.pyplot(fig)

with tab2:
  #tab B를 누르면 표시될 내용
  st.write('hi_hi_new')

