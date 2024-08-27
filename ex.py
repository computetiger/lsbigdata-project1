import pandas as pd
df=pd.read_clipboard() # 외부에서 데이터프레임이나 엑셀을 복사 (ctrl+c)한 것을 그대로 붙여넣기 방법
df=df.sort_values("성별",ignore_index=True)
df_f=df.iloc[:13,:]
df_m=df.iloc[13:,]
df_f=df_f.sample(frac=1).reset_index(drop=True)
df_m=df_m.sample(frac=1).reset_index(drop=True)

df_1=pd.concat([df_f.iloc[:7,],df_m.iloc[:5,]])
df_2=pd.concat([df_f.iloc[7:,],df_m.iloc[5:,]])
