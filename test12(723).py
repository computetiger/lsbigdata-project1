import numpy as np    
np.unique((np.arange(33)-16)**2)
#
a=((np.arange(33)-16)**2)*(2/33)
a.mean()
#
x=np.arange(33)
(x**2).mean()-(x.mean())**2
#
np.unique((x-16)**2)*(2/33)

len(np.unique((x-16)**2)) #17

##
#새로운 확률변수
#가질 수 있는 값: 0,1,2
#20%-0 30%-2 50% -1
def Y(num,p,g):
    x=np.random.rand(num)
    return np.where((x>p),1,np.where((x<g),0,2))
Y(num=10000,p=0.5,g=0.2).mean()

P(x): [0: 1/6, 1:, 2/6, 2: 2/6, 3: 1/6]

import numpy as np
values=np.array(4)
probab=np.array([1/6,2/6,2/6,1/6])

ex2=np.sum(values*probab)**2
e2x=np.sum((values**2)*probab)
e2x-ex2

values1=np.array(99)
probab1=np.arange(0,51/2500,1/2500)
probab2=np.arange(49/2500,0,-1/2500)
probab=np.concatenate((probab1,probab2))
len(probab)
ex2=np.sum(values*probab)**2
e2x=np.sum(((values)**2)*probab)
e2x-ex2

##### 이런 식으로도 쓸 수 있구나 
import numpy as np
values=np.arange(0,7,2)
probab=np.array([1/6,2/6,2/6,1/6])
data[:,0]
data=np.column_stack((values,probab))
pmean=np.sum(data[:,0]*data[:,1])**2
pvar=np.sum((data[:,0]**2)*data[:,1])
pvar-pmean


ex2=np.sum(values*probab)**2
e2x=np.sum((values**2)*probab)
e2x-ex2
###
np.sqrt(4)
