# lec 6 행
import numpy as np    
matrix=np.column_stack((np.arange(1,5),
                        np.arange(12,16)))
print("행렬: \n",matrix)
print("행렬의 크기:", matrix.shape)


y=np.zeros((2,2))
print("빈 행렬:\n",y)
#-1을 통해서 자동으로 결정할 수 있음. 자동으로 올바른 크기 결정. 
np.arange(1,7).reshape((2,-1))

np.arange(1,7).reshape((2,3))


np.random.randint(1,100,50).reshape(5,10)

np.arange(1,7).reshape((2,3),order='C')
np.arange(1,7).reshape((2,3),order='F')

met_a=np.arange(1,21).reshape((4,5),order="C")
met_a.cumsum(axis=0)
met_a.cumsum(axis=1)



met_a[0,0]
met_a[1,1]
met_a[2,3]
met_a[:2,3]
met_a[1:3,1:4]

mat_b=np.arange(1,101).reshape((20,-1))
mat_b[1::2]
mat_b[[3,4,5,6],4]

x=np.arange(1,11).reshape(5,2)*2
x[[True, True, False, False, True],0]
x[:,1]
x[:,1].reshape(-1,1) # 무슨의미? 
x[:,(1,)] # 튜플로 전달해도 2차원으로 나옴옴 
x[:,[1]] 
x[:,1:2]
# 2행의 데이터 중 7으로 나누어 떨어지는 것. 
mat_b[(mat_b[:,1]%7==0),:]
# 사진은 행렬이다. 
import matplotlib.pyplot as plt
np.random.seed(2024)
img1=np.random.rand(3,3)
print("이미지 행렬 img: \n", img1)
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

!pip install imageio
import imageio
import numpy as np
# 이미지 읽기
jelly = imageio.imread("jelly.png")
jelly[:,:,0].shape
jelly[:,:,0].T.shape

plt.imshow(jelly)
plt.imshow(jelly[:,:,0]) # R
plt.imshow(jelly[:,:,1]) # G
plt.imshow(jelly[:,:,2]) # B
plt.imshow(jelly[:,:,3]) # 투명도 
plt.axis('off') #축 정보 없애기
plt.show()
plt.clf()

mat1=np.arange(1,7).reshape(2,3)
mat2=np.arange(7,13).reshape(2,3)

my_array=np.array([mat1,mat2]) #erge, concat과 다른점: 이것은 차원이 늘어난다.
my_array.shape

my_array2=np.array([my_array]*2)
# = np.array([my_array,my_array])
my_array2.shape
my_array2[0][0][0][0]

filt_array=my_array[0,1,1:3]

mat_x=np.arange(1,101).reshape(10,5,2)
mat_x=np.arange(1,101).reshape(5,5,4)
mat_x=np.arange(1,101).reshape(4,5,-1)
len(mat_x)

a=np.array([[1,2,3],[4,5,6]])
a.sum()
a.mean()
a.max()
a.min()
a.std()
a.var()**(1/2)
a.cumsum()
a.cumprod()
a.argmax()
a.argmin()
a.reshape(3,-1)
----

mat_b=np.random.randint(0,100,50).reshape((5,-1))
mat_b
mat_b.mean()
mat_b.max(axis=0) #행별 가장 큰 수는? 
mat_b.max()
mat_b.max(axis=1) #열별 가장 큰 수는? 

mat_b.cumsum(axis=0) #행별 누적합
mat_b.cumsum(axis=1) #열별 누적합
mat_b.sum()

a=np.array([1,3,2,5])
a.cumsum()
a.cumprod()


jelly[:,:,1]
jelly[:,:,2]
jelly[:,:,3]
# 1차원 배열로 변환, clip
c = np.array([[1, 2, 3], [4, 5, 6]])
print("1차원 배열:\n", c.flatten())

d = np.array([1, 2, 3, 4, 5])
print("클립된 배열:", d.clip(2, 4))



