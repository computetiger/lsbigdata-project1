!pip install pandas
!pip install numpy
import pandas as pd
import numpy as np

df=pd.read_csv('data/house/train.csv')
df_saleprice=df.SalePrice

