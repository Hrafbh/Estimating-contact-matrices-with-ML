import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from numpy import linalg as LA

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

from scipy.integrate import solve_ivp


#construction du dataset d'entrainement , input et output 
#input : pour chaque instance , on enrigistre les S,E,I,R pour chaque catégorie d'age
#output : pour chaque instance , on la matrice de contact la matrice de contact

C=[]
for i in range(120,150):
    file_loc=r"E:\Hraf\S8\Projet à Enjeux\Projet Enjeux\matrix_contact_177country\data_contact_others.xls"
    wkb=xlrd.open_workbook(file_loc)
    sheet=wkb.sheet_by_index(i)
    M=np.zeros((16,16))
    for col in range (sheet.ncols):
       for row in range (sheet.nrows-1):    
          M[row][col]=sheet.cell_value(row+1,col)

    N = np.zeros((Nc,1))       
    for kk in range(Nc):
            N[kk] = sum(yinit[4*kk-4:4*kk-1])

    M= M/(N.T)  #équivalent à .T
    tol = 1e-6
    #options = {'atol': tol, 'rtol': tol, 'mxstep': 5000, 'printmessg': False}
    #t = np.linspace(0, 360, 1000)
    y0 = yinit.flatten()
    #A = odeint(lambda y, t : seir1, y0, t, args=(beta, sigma, gamma, alpha, C.T), **options)
    sol = solve_ivp(seir1, tspan, y0, args = (beta, sigma, gamma, alpha, M), method='RK45', rtol=1e-6, atol=1e-6, max_step=0.5)
    A = sol.y
    t = sol.t
    import time
    A = A.T
    start_time = time.process_time()
    end_time = time.process_time()
    execution_time = end_time - start_time
    print("Temps d'exécution :", execution_time)
    t_fin = len(t) 


    S=[]
    E=[]
    I=[]
    R=[]

    X=[]
    
    L=[]
    

    for i in range(16):
        
        S=[LA.norm(A[:100,4*(i+1)-4]),LA.norm(A[100:200,4*(i+1)-4]),LA.norm(A[200:300,4*(i+1)-4]),LA.norm(A[300:400,4*(i+1)-4]),LA.norm(A[400:500,4*(i+1)-4]),LA.norm(A[500:600,4*(i+1)-4]),LA.norm(A[600:700,4*(i+1)-4]),LA.norm(A[700:800,4*(i+1)-4])]
        E=[LA.norm(A[:100,4*(i+1)-3]),LA.norm(A[100:200,4*(i+1)-3]),LA.norm(A[200:300,4*(i+1)-3]),LA.norm(A[300:400,4*(i+1)-3]),LA.norm(A[400:500,4*(i+1)-3]),LA.norm(A[500:600,4*(i+1)-3]),LA.norm(A[600:700,4*(i+1)-3]),LA.norm(A[700:800,4*(i+1)-3])]
        I=[LA.norm(A[:100,4*(i+1)-2]),LA.norm(A[100:200,4*(i+1)-2]),LA.norm(A[200:300,4*(i+1)-2]),LA.norm(A[300:400,4*(i+1)-2]),LA.norm(A[400:500,4*(i+1)-2]),LA.norm(A[500:600,4*(i+1)-2]),LA.norm(A[600:700,4*(i+1)-2]),LA.norm(A[700:800,4*(i+1)-2])]
        R=[LA.norm(A[:100,4*(i+1)-1]),LA.norm(A[100:200,4*(i+1)-1]),LA.norm(A[200:300,4*(i+1)-1]),LA.norm(A[300:400,4*(i+1)-1]),LA.norm(A[400:500,4*(i+1)-1]),LA.norm(A[500:600,4*(i+1)-1]),LA.norm(A[600:700,4*(i+1)-1]),LA.norm(A[700:800,4*(i+1)-1])]


        print(len(S),len(E),len(I),len(R))

        X=X+S+E+I+R

        # X=[S1,E1,I1,R1,.......,Si,Ei,Ii,Ri,.......Sn,En,In,Rn]  avec Si est le vecteur [Si] , X est de taille 64

        


        

    
    C.append(X)    #on stocke les données de chaque instance dans une liste C

data=pd.read_csv(r"E:\Hraf\S8\Projet à Enjeux\Projet Enjeux\source_data\synthetic_contacts_2021.csv")
data=data.to_numpy()

df = pd.DataFrame(C,index=[data[((256-(16*(1+10)))+10)+256*(5*i)][0] for i in range(120,150)]) 
df.to_excel('input_seri_others.xlsx', sheet_name='dataset',header=[f'feature_{i+1}' for i in range(512)])#run this code to build the dataset