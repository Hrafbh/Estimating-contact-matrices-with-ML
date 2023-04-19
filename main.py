import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.io
from numpy import linalg as LA
import pandas as pd

Nc = 16
vc = np.arange(1, Nc+1)
print(vc.shape)
tspan = [0, 400]
sigma = 1/14
gamma = 6.19e-04
beta = 0.0962

alpha = np.array([0.0050, 0.0009, 0.0010, 0.0045, 0.0090, 0.0150, 0.0350, 0.0600, 0.0700, 0.0850, 0.3000, 0.5500, 0.6500, 1.0000, 2.0000, 6.0000])

yinit = np.zeros((4*Nc,1))
Nb_sus = 3.691e7 # = 3.691.10^7
Pourc = np.ones((1, Nc))
Pourc = np.array([9, 9.3, 8.5, 8, 7.8, 8.1, 7.7, 7.5, 6.4, 5.6, 5.2, 4.9, 4.2, 3.2, 1.8, 1.4]) / 100

for mm in range(1,Nc+1):
    yinit[4*mm-4] = Nb_sus * Pourc[mm-1]


for i in range(Nc) :        
    yinit[(4*vc-4)[i]] = yinit[4*8-4]-1
    yinit[(4*vc-3)[i]]= 0 # E_0
    yinit[(4*vc-2)[i]] = 0  # I_0
    yinit[4*8-2] = 1  # I_0 I_8=1
    yinit[(4*vc-1)[i]] = 0 # R_0

#C = np.zeros((16,16))
CH1 = scipy.io.loadmat("E:\Hraf\S8\Projet à Enjeux\Python\matrix_house.mat")
CH1 = np.array(CH1['matrix_house'])
CH2 = scipy.io.loadmat("E:\Hraf\S8\Projet à Enjeux\Python\matrix_work.mat")
CH2 = np.array(CH2['matrix_work'])
CH3 = scipy.io.loadmat("E:\Hraf\S8\Projet à Enjeux\Python\matrix_school.mat")
CH3 = np.array(CH3['matrix_school'])
CH4 = scipy.io.loadmat("E:\Hraf\S8\Projet à Enjeux\Python\matrix_public.mat")
CH4 = np.array(CH4['matrix_public'])
CH5 = scipy.io.loadmat("E:\Hraf\S8\Projet à Enjeux\Python\matrix_shop.mat")
CH5 = np.array(CH5['matrix_shop'])
C = CH1 + CH2 + CH3 + CH4 + CH5
#options = {'atol': tol, 'rtol': tol, 'mxstep': 5, 'printmessg': False}
def seir1(t,y, beta, sigma, gamma, alpha, C):
    
    Nc = 16
    vc = np.arange(1, Nc+1)
    
    yprime = np.zeros((4*Nc,1))
    vc = np.arange(1, Nc+1)
    yprime = np.zeros((4*Nc,))
    yprime[4*vc-3-1] = -beta*y[4*vc-3-1]*(C @ y[4*vc-1-1])
    yprime[4*vc-2-1] = beta*y[4*vc-3-1]*(C @ y[4*vc-1-1]) - sigma*y[4*vc-2-1]
    yprime[4*vc-1-1] = sigma*y[4*vc-2-1] - gamma*y[4*vc-1-1] - alpha*y[4*vc-1-1]
    yprime[4*vc-1] = gamma*y[4*vc-1-1]
    return yprime


#from SEIR import seir1
from scipy.integrate import solve_ivp

N = np.zeros((Nc,1))
for kk in range(Nc):
    N[kk] = sum(yinit[4*kk-4:4*kk-1])
print(C.shape)
C = C / (N.T)  #équivalent à .T
print(C.shape)
tol = 1e-6
#création de structure d'options pour la fonction d'intégration numérique ODE (Ordinary Differential Equation)
#options = {'atol': tol, 'rtol': tol, 'mxstep': 5000, 'printmessg': False}
#t = np.linspace(0, 360, 1000)
y0 = yinit.flatten()
#A = odeint(lambda y, t : seir1, y0, t, args=(beta, sigma, gamma, alpha, C.T), **options)
sol = solve_ivp(seir1, tspan, y0, args = (beta, sigma, gamma, alpha, C), method='RK45', rtol=1e-6, atol=1e-6, max_step=0.5)
A = sol.y
t = sol.t
import time
A = A.T
start_time = time.process_time()

# Code à mesurer

end_time = time.process_time()
execution_time = end_time - start_time

print("Temps d'exécution :", execution_time)
t_fin = len(t) 
print(len(t))
print(A.shape)

S=[]
E=[]
I=[]
R=[]
X=[]

for i in range(16):
        
    S=[LA.norm(A[:100,4*(i+1)-4]),LA.norm(A[100:200,4*(i+1)-4]),LA.norm(A[200:300,4*(i+1)-4]),LA.norm(A[300:400,4*(i+1)-4]),LA.norm(A[400:500,4*(i+1)-4]),LA.norm(A[500:600,4*(i+1)-4]),LA.norm(A[600:700,4*(i+1)-4]),LA.norm(A[700:800,4*(i+1)-4])]
    E=[LA.norm(A[:100,4*(i+1)-3]),LA.norm(A[100:200,4*(i+1)-3]),LA.norm(A[200:300,4*(i+1)-3]),LA.norm(A[300:400,4*(i+1)-3]),LA.norm(A[400:500,4*(i+1)-3]),LA.norm(A[500:600,4*(i+1)-3]),LA.norm(A[600:700,4*(i+1)-3]),LA.norm(A[700:800,4*(i+1)-3])]
    I=[LA.norm(A[:100,4*(i+1)-2]),LA.norm(A[100:200,4*(i+1)-2]),LA.norm(A[200:300,4*(i+1)-2]),LA.norm(A[300:400,4*(i+1)-2]),LA.norm(A[400:500,4*(i+1)-2]),LA.norm(A[500:600,4*(i+1)-2]),LA.norm(A[600:700,4*(i+1)-2]),LA.norm(A[700:800,4*(i+1)-2])]
    R=[LA.norm(A[:100,4*(i+1)-1]),LA.norm(A[100:200,4*(i+1)-1]),LA.norm(A[200:300,4*(i+1)-1]),LA.norm(A[300:400,4*(i+1)-1]),LA.norm(A[400:500,4*(i+1)-1]),LA.norm(A[500:600,4*(i+1)-1]),LA.norm(A[600:700,4*(i+1)-1]),LA.norm(A[700:800,4*(i+1)-1])]


    print(len(S),len(E),len(I),len(R))

    X=X+S+E+I+R

df = pd.DataFrame(X) 
df.to_excel('input_seri_test.xlsx', sheet_name='dataset')







y = np.zeros((t_fin, 4))

y[:, 0] = np.sum(A[:, 4*vc-3-1], axis=1)
y[:, 1] = np.sum(A[:, 4*vc-2-1], axis=1)
y[:, 2] = np.sum(A[:, 4*vc-1-1], axis=1)
y[:, 3] = np.sum(A[:, 4*vc-1], axis=1)

plt.plot(t, y[:,0], 'b-', t, y[:,1], 'k', t, y[:,2], 'r', t, y[:,3], 'g')
plt.legend(['S: susceptible', 'E: exposed', 'I: infectious', 'R: recovered'])
plt.xlabel('time (days)')
plt.ylabel('Number of persons')
plt.gca().yaxis.set_major_formatter('{:.0f}'.format)
plt.title('Evolution of classes')

for ii in range(1, 6+1):
    plt.figure(3)
    plt.subplot(3, 2, ii)
    plt.plot(t, A[:,4*ii-4], 'b-', t, A[:,4*ii-3], 'k', t, A[:,4*ii-2], 'r', t, A[:,4*ii-1], 'g')
    plt.legend(['S: susceptible', 'E: exposed', 'I: infectious', 'R: recovered'])
    plt.xlabel('time (days)')
    plt.ylabel('Number of persons')
    plt.gca().yaxis.set_major_formatter('{:.0f}'.format)
    plt.title('Evolution of class {}'.format(ii))
    



print(y[:, 0])