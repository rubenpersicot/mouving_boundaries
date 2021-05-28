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

@njit
def pressureGradContrib(rho_i,rho_j,P_i,P_j,dwdr,er,m):
    '''pressureGrad : Pressure contribution to the momentum equation
            dF = -G_i(p_j)/rho_i
        pressureGradContrib(rho_i,rho_j,P_i,P_j,dwdr,er,m) 
        returns 
        dF of dim size(er)
    '''
    dF = np.zeros(np.shape(er))
    # COMPLETE HERE
    dF[:,0] = -(m*(P_i/rho_i**2+P_j/(rho_j**2))*dwdr)*er[:,0];
    dF[:,1] = -(m*(P_i/rho_i**2+P_j/(rho_j**2))*dwdr)*er[:,1];
    # END
    return dF

@njit
def velocityDivContrib(rVel,rPos,dwdr,er,m):
    '''velocityDiv : velocity contribution to the continuity equation
            dV = m*rVel.rPos*\nablar(w) /|r|
       returns 
            dV of dim size(er)
    '''
    dV = np.zeros(np.shape(er))
    veldotpos = rVel[:,0]*rPos[:,0]+rVel[:,1]*rPos[:,1]
    rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
    # COMPLETE HERE
    dV = m*veldotpos*dwdr/rNorm
    # END
    return dV

@njit
def pressureInterpolationContrib(rho_j, P_j, vol_j,rPos,w_ij,grav):
    '''PressureInterpolation :  interpolate the pressure based
                Cf Adami, Hu & Adams, 2012
    '''
    pressInt = np.zeros(np.shape(rPos))
    #COMPLETE HERE
    pressInt = (P_j + rho_j*(rPos@grav.transpose()))*vol_j*w_ij
    #END
    return pressInt

@njit
def shepardContrib(vol_j,w_ij):
    ''' 
    Shepard
    '''
    shep = np.zeros(np.shape(vol_j))
    #COMPLETE HERE
    shep = vol_j*w_ij
    #END
    return shep

@njit
def artViscContrib(mu,rho_i, rho_j,dwdr,rVel,rPos,m,dr,eps):
    '''
    Artificial viscosity
    '''
    veldotpos = rVel[:,0]*rPos[:,0]+rVel[:,1]*rPos[:,1]
    rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
    mu[veldotpos>0] = 0
    F = np.zeros_like(rPos)
    F[:,0] = m/rho_i*(mu/rho_j)*(veldotpos/(rNorm**2+eps*dr**2))*dwdr*rPos[:,0]/rNorm
    F[:,1] = m/rho_i*(mu/rho_j)*(veldotpos/(rNorm**2+eps*dr**2))*dwdr*rPos[:,1]/rNorm
    return F

@njit
def MonaghanViscContrib(mu,rho_i, rho_j,dwdr,rVel,rPos,m):
    F = np.zeros_like(rPos)
    #TODO : COMPLETE HERE
    veldotpos = rVel[:,0]*rPos[:,0]+rVel[:,1]*rPos[:,1]
    rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
    F[:,0] = 2*(2+2)*m*mu/rho_i/rho_j*veldotpos/rNorm**2*dwdr*rPos[:,0]/rNorm
    F[:,1] = 2*(2+2)*m*mu/rho_i/rho_j*veldotpos/rNorm**2*dwdr*rPos[:,1]/rNorm
    #END
    return F

@njit
def MorrisViscContrib(mu,rho_i, rho_j,dwdr,rVel,rPos,m):
    F = np.zeros_like(rPos)
    #TODO : COMPLETE HERE
    rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
    F[:,0] = 2*m*(mu/(rho_i*rho_j))*dwdr/rNorm*rVel[:,0]
    F[:,1] = 2*m*(mu/(rho_i*rho_j))*dwdr/rNorm*rVel[:,1]
    #END
    return F

@njit
def FFluidSolidContrib(pi, pj, mu, rho_i, rho_j,dwdr,rVel,rPos,m,ms):
    F = np.zeros_like(rPos)
    #TODO : COMPLETE HERE
    rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
    veldotpos = rVel[:,0]*rPos[:,0]+rVel[:,1]*rPos[:,1]
    F[:,0] = (-(pi+pj)+mu*2*(2+2)*(veldotpos/rNorm**2))*m/rho_i*ms/rho_j*dwdr*rPos[:,0]/rNorm
    F[:,1] = (-(pi+pj)+mu*2*(2+2)*(veldotpos/rNorm**2))*m/rho_i*ms/rho_j*dwdr*rPos[:,1]/rNorm
    #END
    return F


