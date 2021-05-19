###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
#Authors :  R. Carmigniani & D. Violeau
#Version : SPyH
#Contact : remi.carmigniani@enpc.fr
###########################################################################
from src.spyh import *
from src.contrib import *
from src.state import *
import numpy as np
def q1b(g,mu):
    print('Your values :\n\t g = %2.2f\n\t mu = %2.2f'%(g[0],mu))
    if  g[0] == 4 and mu==500:
        print('1-b) All good!')
        check=True
    else:
        print('1-b) There is an error...')
        check=False
def q2a():
    m=1
    mu=1
    rho_i=1000
    rho_j = np.array([1000,1001,1003])
    dwdr = np.array([1,2,3])
    rVel = np.array([[0,1],[0.5,0.2],[0.1,0.23]])
    rPos = np.array([[0.04,0.04],[1,1],[0.2,0.24]])
    FMor = MorrisViscContrib(mu,rho_i, rho_j,dwdr,rVel,rPos,m)
    FMorExp = np.array([[0.00000000e+00, 3.53553391e-05],
       [1.41280076e-06, 5.65120305e-07],
       [1.91480877e-06, 4.40406018e-06]])
    FMon = MonaghanViscContrib(mu,rho_i, rho_j,dwdr,rVel,rPos,m)
    FMonExp = np.array([[7.07106781e-05, 7.07106781e-05],
       [3.95584213e-06, 3.95584213e-06],
       [1.18027557e-05, 1.41633069e-05]])
    if np.linalg.norm(FMor-FMorExp)<10**-12:
        print('Morris is correctly implemented')
    else :
        print('Error in Morris...')
    if np.linalg.norm(FMon-FMonExp)<10**-12:
        print('Monaghan is correctly implemented')
    else :
        print('Error in Monaghan...')