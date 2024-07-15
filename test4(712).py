#0712
a=[1,2,3]
b=a
b

a[1]=4
b

id(a)
id(b)
#a=b는 할당(id주소 동일), a=b[:] 은 복붙붙
a=[1,2,3]
b=a[:]
b=a.copy()

a[1]=4
a
b
id(a)
id(b)
#
c=[1,2,3]
d=c.copy()
c[1]=10
c
d

#math : 기본적으로 내장된 모듈, 설치는 안해도 되나 나타나는 건 아니어서 import해줘야함함
import math
x=4
math.sqrt(x)

val=math.sqrt(16)
print(val)

exp=math.exp(5)
print(exp)

logg=math.log(10,10)
print(logg)

fact=math.factorial(5)
print(fact)

sin=math. sin(math.radians(90))
cos=math. cos(math.radians(125))
tan=math. tan(math.radians(35))

print(f"""
sin의 값은 {sin} 입니다. 
cos의 값은 {cos} 입니다. 
tan의 값은 {tan} 입니다. 
""")
#확률밀도함수 구해보기
def my_normal_pdf(x, mu, sigma):
  part_1=1/(sigma*math.sqrt(2*math.pi))
  part_2=math.exp(-0.5*(((x-mu)/sigma)**2))
  return part_1*part_2
my_normal_pdf(3,3,1)

#연습 
def my_fun(x,y,z):
  result=(x**2+math.sqrt(y)+math.sin(z))* math.exp(x)
  return result
my_fun(2,9,math.pi/2)

#연습
def fun_g(x):
  return math.cos(x)+math.sin(x)*math.exp(x)
fun_g(math.pi)

# numpy 실습
#!pip install numpy
import numpy as np
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

type(a)

a[3]
#int
type(a[2:])
a[1:4]
#vector 생성하기 
x= np.empty(3)
print(x)

x[0]=1
x[1]=3
x[2]=5
type(x[2])

#
vec1=np.array(1,2,3,4,5)
vec1=np.arange(100)
vec1=np.arange(1,100.1,0.5)
vec1

#linspace: 첫값부터 끝값제외하고 까지 마지막 수 갯수만큼 배열 생성성
l_space1=np.linspace(0,100,100)
l_space1
#endpoint=False: 끝값 제외하고 5개의 숫자 배열 생성
l_space2=np.linspace(0,1,5, endpoint=False)
l_space2

#repeat: var 를 5 번 반복
var=np.arange(5) #5보다 작은 정수까지의 배열
np.repeat(var,5)

var=np.arange(0,-100,-1)
var

#
vec3=np.arange(0,100)
vec3

#
var=np.arange(5)
np.repeat(var,3)
np.tile(var,3)

#
var+1
var*2
var*var

#vector의 크기가 같아야 한다. 
varr=np.arange(4)
var*varr

#기술통계 
sum(var)
max(var)
min(var)

#35672 이하 홀수들의 합
sum(np.arange(1,35672,2))
np.arange(1,35672,2).sum()

a=np.array([1,2,3,4,5])
len(a)
a.shape
##
a=np.array([1,2,3,4,5,np.nan])
len(a)
a.shape
a.size

#
b=np.array([1,2,3],[4,5,6])

length=len(b) #3 : 첫 번째 차원의 길이
shape=b.shape #3 : 각 차원의 크기
size=b.size #3 : 전체 요소의 개수

#
a=np.array([1,2])
b=np.array([5,6,7,8])
np.tile(a,2)+b
np.repeat(a,2)+b

#bool : 비교 연산자 
b=np.array([5,6,7,8])
b==6

#35672 보다 작은 수 중에서 7로 나눠서 나머지가3인 숫자들의 개수
#1
n=np.arange(3,35672,7)
len(n)
#2
sum((np.arange(1,35672)%7)==3)
n=(np.arange(3,35672)%7==3)
sum(n) # n가체가 true, false 로 반환. true=1, false=0 이기에 sum이 가능하다. 



a=np.array([1,2,3],[4,5,6])
b=np.array([5,6,7],[8,9,10])
