#list
numbers=[[1,2,3,4,5],[6,7,8,9,10]]
numbers[0][2]

import numpy as np
#list: 대괄호로 쌓여있다. 
# 넣고 싶은 수식표현을 x를 사용해서 표현
# for in 을 사용해 원소정보 제공 
a=[x**2 for x in range(10)]

list(range(10))

my_list=[x**3 for x in [3,5,2,15]]
np_list=[x**3 for x in np.array([3,5,2,15])]
type(np_list)

#Pandas Serise
#!pip install pandas
import pandas as pd
df=pd.read_csv('data/exam.csv')
df
#
list1=[1,2,3,4]
list1*2
list2=[5,6,7,8]
list2+list1
#
numbers=[5,2,3]
num_list=[x*2 for x in numbers]
num_list=[lambda x: x**2 ,numbers]
type(num_list)
a=[x for x in numbers for _ in range(4)] # 
a,_,b=1,2,3
#
5+4
_+6
_=None
del _

for x in numbers:
    for y in range(4):
        print(x,":",y)
        
# 리스트를 하나 만들어서 for 루프를 사용해 2,4,6,8... 20의 수를 채워 넣어 보기. 
my_list=[]
for i in range(1,11):
   my_list.append(2*i)
   
my_list
mylist=list(range(4))

mylist=[0]*10
for i in range(10):
    mylist[i]=2*(i+1)
mylist

mylist_b=[2,4,6,8,10,12,14,16,18]
mylist=[0]*10
a=[]
for i in range(len(mylist_b)):
    a.append(i)
#
mylist_b=[2,4,6,80,10,12,24,35,23,20,100]
mylist=[0]*10

for i in mylist_b:
    if i==0:
        mylist.append(mylist_b[i])
    elif (i%2)==0:
        mylist.append(mylist_b[i])
    else:
        continue
    
mylist_b=[2,4,6,80,10,12,24,35,23,20,100]
mylist=[0]*5

for i in range(len(mylist)):
    if i % 2 == 0:
        mylist[i]=mylist_b[i]
    
# list comprehenshion 으로 바꾸는 방법. 바깥은 무조건 대괄호로 붂어줌: 리스트 반환하기 위해서. 
# for루프의 : 는 생략한다. 실행부분을 먼저 써준다. 

[i*2 for i in range(1,11)]
[x for x in numbers]

for i in range(1,5):
    for j in range(1,3):
        print(f"{i}*{j}={i*j}")
    print("----------")
    
#원소 체크
mlist=[]
fruits=["apple","banana","cherry","banana"]
"banana" in fruits

for i in fruits: 
    mlist.append(i=="banana")

#true인 값의 인덱스 추출
for i in fruits: 
    if i =="banana":
        print(fruits.index(i))
#
for i in fruits: 
    if i =="banana":
        print(i)

[i,j for i in numbers for j in range(4)]
    
#
fruits=["apple","banana","cherry","banana"]
import numpy as np
fruits=np.array(fruits)
int(np.where(fruits=="banana")[0][0])
#mylist.index(i)
fruits=["apple","banana","cherry","banana"]
fruits.reverse()
fruits.insert(2,"fake")

fruits.insert(2,"apple")

print(fruits.pop(1))
print(fruits.remove("apple"))# 첫 번째 원소만 삭제 

# (불리언 마스크)논리형 벡터 생성
t_remove=np.array(["banana","apple"])
mask= ~np.isin(fruits, t_remove) # np.array로 T or F 값 전달 
mask= ~np.isin(fruits, ["banana","apple"]) # list 로 전달해도 가능


#

mylist_b=["aa","bb","cc","dd","ff","gg","rr","qq","zz"]
xx = []
for i in range(len(mylist_b)):
    if i % 2 == 0:
        xx.append(mylist_b[i])
        xx
print("-------------")
#

mylist_b=["aa","bb","cc","dd","ff","gg","rr","qq","zz"]
xx = []
for i in range(len(mylist_b)):
    if i % 2 == 0:
        xx=mylist_b[i]
        xx
print("-------------")
xx
