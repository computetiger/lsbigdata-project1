import numpy as np
a=np.array([1,2,3,4]).reshape((2,2), order="F")
b=np.array([5, 6, 7, 8]).reshape(2,-1)
# vector 의 내적 
a.dot(b)
a @ b
##
c=np.array([1,2,1,0,2,3]).reshape((2,3), order="F")
d=np.array([1,0,-1,1,2,3]).reshape(3,-1)
# vector 의 내적 
c.dot(d)
c @ d
# Q2 
np.eye(3)
a=np.array([3,5,7,2,4,9,3,1,0]).reshape(3,3)
d.T
np.eye(3) @ a
#회귀분석 데이터행렬
x=np.array([13,15,12,14,10,11,5,6]).reshape(4,2)
vec1=np.repeat(1,4).reshape(4,1)
matX=np.hstack((vec1,x))
beta_vec=np.array([2,0,1]).reshape(3,1)
matX @ beta_vec

#
y=np.array([20,19,20,12]).reshape(4,1)
(y-matX @ beta_vec).T @ (y-matX @ beta_vec)
# 2 * 2 역행렬 공식
a_inv=(-1/11) * np.array([4,-5,-3,1])
a=np.array()
# 3 by 3 역행렬
a=np.array([-4,-6,2,
            5,-1,3,
            -2,4,-3]).reshape(3,3)
a_inv=np.linalg.inv(a)
np.linalg.det(a)


# 역행렬 없을수도 
s=np.array([1,2,5,
            2,7,6,
            3,6,9]).reshape(3,3)
s_inv=np.linalg.inv(s)

xtx_mat=np.linalg.inv((matX.T@matX))
xty=matX.T @ y
beta_hat=xtx_inv @ 
# beta 구하기 
matX
xtx_inv= np.linalg.inv((matX.T @ matX))
xty=matX.T @ y
beta_hat=xtx_inv @ xty


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(matX[:, 1:], y)

model.intercept_
model.coef_


# minimize, line_ perform. 
from scipy.optimize import minimize
def line_perform(beta):
    beta=np.array(beta).reshape(3,1)
    a=(y-matX @ beta)
    return (a.T @ a)

line_perform([8.55, 5.96, -4.38])
line_perform([3.76, 1.36, 0])

# minimize로 lasso 베타 구하기. 
from scipy.optimize import minimize
def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3,1)
    a=(y-matX @ beta)
    return (a.T @ a) + 3*np.abs(beta).sum()

      
# 초기 추정값
initial_guess=[0,0,0]

line_perform_lasso([8.55, 5.96, -4.38])
line_perform_lasso([3.76, 1.36, 0])
             
result=minimize(line_perform_lasso, initial_guess)

print("최소값", result.fun)
print("최소값을 갖는 x 값", result.x)



# minimize로 ridge 베타 구하기. 
from scipy.optimize import minimize
def line_perform_ridge(beta):
    beta=np.array(beta).reshape(3,1)
    a=(y-matX @ beta)
    return (a.T @ a) + 3*(beta**2).sum()

# 초기 추정값
initial_guess=[0,0,0]
# 릿지
line_perform_ridge([8.55, 5.96, -4.38])
line_perform_ridge([8.14, 0.96, 0])
                   
result=minimize(line_perform_ridge, initial_guess)

print("최소값", result.fun)
print("최소값을 갖는 x 값", result.x)

[8.55, 5.96, -4.38] # 람다 0

[17.44, 0, 0] # 람다 500
# 예측식: y_hat = 17.74 + 0 * x1 + 0 * x2
# 람다를 500으로 설정하면 값이 17.74 가 나오게 

[8.14, 0.96, 0] # 람다 3
# 예측식: y_hat = 8. 14 + 0.96 * x1 + 0 * x2