#균일확률변수 만들기 
import numpy as np

n=0
while n<9:
    np.random.rand(1)
    n+=1
print("end")


def x(i):
    return np.random.rand(i)
x(8)

# 베르누이 확률변수 모수가 p 만들기 
def Y(num,p):
    x=np.random.rand(num)
    return np.where((x<p),1,0)
Y(p=0.5,num=100000000).mean()

#새로운 확률변수
#가질 수 있는 값: 0,1,2
#20%-0 30%-2 50% -1
def Y(num,p,g):
    x=np.random.rand(num)
    return np.where((x>p),1,np.where((x<g),0,2))
Y(num=10000,p=0.5,g=0.2).mean()

#확률을 지정하는경우 , 누적합 사용. 
def Z(p):
    x=np.random.rand(1)
    p_cumsum=p.cumsum()
    return np.where(x<p_cumsum[0],0,np.where(x<p_cumsum[1],1,2))

#7월 19일, 이어서 공부 

p=np.array([0.1,0.6,0.3])
Z(p)

(np.arange(4))* (np.array([1,2,3,4])/6)


