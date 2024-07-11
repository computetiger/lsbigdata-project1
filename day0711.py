# 데이터 타입
x=10.44
print(f"{x}는 {type(x)} 형식입니다.")

print("""
hello
hi
new
""")

print("""\
hello
hi
new\
""")

a='hi'*3
print('반복', a)

a=(1,2,3,4,5)
a[2]
a[:4]

b=(43)
type(b)
c=(43,)
type(c)

def m_m(num):
  return {min(num), max(num)}

result=m_m((1,2,3,4,5,6,7))
type(result)
print('min and max:', result)




a=set()
type(a)
a.add(3)
a

a_list=[10,20,30]
a_tuple=(10,20,30)

a_list[1]=25
a_tuple[1]=25


li=[1,2,3,4,5]
li=tuple(li)
li
###
set_example={'a','b','c'}
#집합을 사용하여 딕셔너리를 생성할 때 키의 순서는 집합에서 키가 꺼내지는 순서에 따라 달라질 수 있습니다.
dfs={key: False for key in set_example}
print("dictionary from set",dfs)


p={
  'name':'Jiwon',
  'age':27,
  'city':'Seoul'}

t={
  'name':'Jiwon',
  'age':(25,27),
  'city':['Seoul','NewYork']
  }

tt=t.get('city')
tt[0]

a=set()
print(a, type(a))
a.add('qq')
a.add('rr')
a.add('qq')
a.remove('qq')
a.discard('qq')
a

#집합 간 연산
a={'banana','apple'}
b={'apple','gum'}
a.union(b)
a.intersection(b)
a.difference(b)
set.intersection(a,b)

num=1.9
int(num)

round(3.45635435,5)
