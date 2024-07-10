"""
terminal 명령어 

dir, ls 현재 파일목록 
cd 해당 디렉토리로 이동
, 현재 폴더 
,, 상위 폴더 
PS : 파워쉘
tab/ shift + tab : 자동완성
cls: 화면 정리 
"""

a=' he says 'hello' to me!'
a

b= "he says 'hello' to me!"
b

#work
a=[1,2,3,4,5]
b=[4,5,6]
a+b

#work
str1="안녕하세요"
str2="제 이름은"
str3="ooo입니다"
str1+' '+str2+' '+str3


#work
a=10
b=3.3

print("a+b =", a+b)
print("a-b =", a-b)
print("a*b =", a*b)
print("a/b =", a/b)
print("a%b =", a%b)
print("a//b =", a//b)
print("a**b =", a**b)

#work 
a='hello'
b='4'
a+b

#work 
print("역슬래쉬를 실행해보자. \n 줄바꿈도 실행해보자")
print("역슬래쉬를 실행해보자.\t줄바꿈도 실행해보자")
print("역슬래쉬를 실행해보자.\"줄바꿈도\" 실행해보자")

print('''
이렇게 해서 
줄바꿈 
해보는거지
''')

#shift + alt + 아래화살표 : 아래로 복사
#ctrl+alt+아래화살표 : 커서 여러개 >> 잘 안되네...
#work
a=10
b=20
print("a==b", a==b)
print("a!=b", a!=b)
print("a>b", a>b)
print("a<b", a<b)
print("a<=b", a<=b)
print("a>=b", a>=b)

#work
age=20
is_adult=age>19
print("성인입니까?", is_adult)


#work
a=((2**4)+(12453//7))%8
b=((9**7)/12)*(36452%253)
a
b

#work 
TRUE=55
true=44
a="True"
b=TRUE
c=true
d=True

#work 
a=True
b=False
print(a or b) 
print(a and b)
print(not a)
print(not b)

True and True
False and False
True or False
False or False

not False
not True

True | True
True+ False
False-False
True*4

True and False : and = mul
True or False : or = plus, 
#단, true or true 는 1인데, 이 경우 '컴퓨터는 1 이상의 값을 1이라 인식, 혹은 0보다 큰값은 1로 인식
#한다고 생각해볼 수 있다. 
#true or false 를 언제 사용할까? 필터링 할때. 


a= True 

b=False 

print(int(a|b))
print(int(b|b))
int(a|a)
int(b|a)


a= True 

b=False 

print(int(a&b))
print(int(b&b))
print(int(a&a))
print(int(b&a))

#복합 대입 연산자 
a=10
a+=10
b=13
b-=5
c=9
c*=3
d=8
d/=2

#str 과 음수 곱셈
str1='hello'
str1*3
str1* (-4)

x=5
~x
bin(~x)

import pydataset
pydataset.data()

#df=pydataset.data("AirPassengers")
import pandas as pd
df=pd.read_csv("sample1.csv")
df
