###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
#Authors :  R. Carmigniani & D. Violeau
#Version : SPyH.0 
#Contact : remi.carmigniani@enpc.fr
###########################################################################
# Some useful imports
import numpy as np
from numba import njit
from src.sphvar import *

def analyticalPoiseuilleFlow(y,Re,t,n=10):
	'''
	return the analytical solution for a sudden start of the Poiseuille Flow
	'''
	z = (y+1)/2
	u = 0*z
	for i in range(n+1):
		u = u+ np.sin((2*i+1)*np.pi*z)/((2*i+1)**3*np.pi**3)*\
			(1-np.exp(-(2*i+1)**2*np.pi**2/(2*Re)*t))
	return 32*u

def analyticalCouetteFlow(y, U, e, t, n=10):
	'''
	return the analytical solution for a sudden start of the Couette Flow
	'''
	z = (1+y)*0.5
	u = 0*z
	for i in range(n+1):
		u = (U+e)*z/e - e
	return u
    