import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')




# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()


#--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x=10; y=10
lstep=0.0005

for i in range(500) : 
    (x, y) = np.array([x, y]) - lstep * np.array([8*x+20*y-23, 20*x+60*y-67])
    plt.scatter(float(x), float(y), color='red', s=5)
print(x,y)
y.min()
plt.xlim(5.5, 6.0)
plt.ylim(-1,0.5)




# x, y의 값을 정의합니다 (-1에서 7까지)
b = np.linspace(-10, 10, 400)
B = np.linspace(-10, 10, 400)
b, B = np.meshgrid(b, B)

# 함수 f(x, y)를 계산합니다.
f = (1-(b+B))**2 + (4-(b+2*B))**2 + (1.5-(b+3*B))**2 +(5-(b+4* B))**2


# 등고선 그래프를 그립니다.
ig = plt.figure()
ax = fig.add_subplot(300, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(b,B,f, cmap='viridis')







plt.figure()
cp = plt.contour(b, B, f, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.
plt.scatter(9,2,color="red",s=20)
for i in range(100):
    x1, y1 = np.array([x1,y1]) -lstep * np.array([2*x1-6,2*y1-8])
    plt.scatter(float(x1),float(y1), color="red", s=25)
print(x1,y1)
#plt.scatter(x1,y1,color="blue",s=20)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()
#------------------------------
x=10
lstep=np.arange(100,0,-1)*0.01

len(lstep)

for i in range(100):
    x-=lstep[i]*(2*x)
print(x)
#----------------------------

# x, y의 값을 정의합니다 (-1에서 7까지)
B0 = np.linspace(-10, 10, 400)
B1 = np.linspace(-10, 10, 400)
B0, B1 = np.meshgrid(B0, B1)

# 함수 f(x, y)를 계산합니다.
z = (1-(B0+B1))**2 + (4-(B0+2*B1))**2 + (1.5-(B0+3*B1))**2 + (5-(B0+4*B1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(B0, B1, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# f(B0, B1) = (1-(B0+B1))**2 + (4-(B0+2*B1))**2 + (1.5-(B0+3*B1))**2 + (5-(B0+4*B1))**@
B0 = 10
B1 = 10
delta = 0.0001
for i in range(100000):
    B0, B1 = np.array([B0, B1]) - delta * np.array([8*B0 + 20*B1 -23, 20*B0 + 60*B1 -67])
    plt.scatter(B0, B1, color = 'red', s=7)
print(B0, B1)

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df=pd.DataFrame({
    "x": np.array([1,2,3,4]),
    "y": np.array([1,4,1.5,5])
})
model=LinearRegression()
model.fit(df[["x"]],df["y"])
model.intercept_, model.coef_
#------------------------------
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df=pd.DataFrame({
    "y": np.array([13,15,17,15,16]),
    "x1": np.array([16,20,22,18,17]),
    "x2":np.array([7,7,5,6,7])
})
model=LinearRegression()
model.fit(df[["x1","x2"]],df["y"])
model.intercept_, model.coef_
model.coef_[0] * 15 + model.intercept_

