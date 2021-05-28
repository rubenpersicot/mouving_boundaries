###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
#Authors :  yohan lanier and ruben persicot
#Version : SPyH
###########################################################################
# Kernel Wendland
import numpy as np
from numba import njit
from src.sphvar import *
from src.state import *
from src.contrib import *

@njit
def computeCenterOfMass(part, n):
    '''
    -part : ....
    - n : number of solid particle
    
    '''
    infoTab = part[:,INFO]
    OG = np.array([1/n*np.sum(part[infoTab == MOBILESOLID] [:,POS[0]]),1/n*np.sum(part[infoTab == MOBILESOLID] [:,POS[1]])])
    return OG

@njit   
def computeForcesFluidSolid(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,mu,d=2):
    '''
        Compute the forces using Morris viscous forces
        and the RHS for the continuity equation
        input :
            - partMOBILESOLID : True if a MOBILESOLID particle
            - partSPID :  particle space ID
            - partPos : particle position
            - partVel : particle velocity
            - partRho : particle density
            - listNeibSpace : list of the particle influencing a particle in a space spId
            - aW : constant of kernel
            - h : smoothing length
            - m : particle mass
            - B : state constant
            - rhoF : reference density
            - gamma : polytropic gas constant
            - grav : gravitational acceleration
            - mu : viscosity
            - d : dimension
        return :
            - forces : table of the particle forces
    '''
    forces = np.zeros_like(partVel)
    drhodt = np.zeros_like(partRho)
    nPart = len(partMOBILESOLID)
    for i in range(nPart):
        if partpartMOBILESOLID[i]:
            spid_i = int(partSPID[i])
            #list neib
            listnb = listNeibSpace[spid_i,:]
            listnb = listnb[listnb>-1]
            listnb = listnb[listnb!=i] #no self contribution
            #---------------------------------------------------
            #removing non fluid particle from the neighbours list
            listFluidNb = []
            for i in range(len(listnb)):
                if part[listnb[i], INFO] == FLUID : 
                listFluidNb.append(listnb[i])
            listFluidNb=np.array(listFluidNb)
            #---------------------------------------------------
            #Position, norm  and er
            rPos = partPos[listFluidNb][:]-partPos[i,:]
            rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
            q=rNorm/h
            dwdr = Fw(q,aW,h)
            er = np.zeros_like(rPos)
            er[:,0] = rPos[:,0]/rNorm
            er[:,1] = rPos[:,1]/rNorm
            #velocity
            v_s = partVel[i,:]
            #Continuity Velocity :  the velocity is the true wall velocity 
            v_f = partVel[listFluidNb][:]
            #v_j[partFLUID[listnb]==False][:,0]= 0 #set wall velocity to true vel
            #v_j[partFLUID[listnb]==False][:,1]= 0 #set wall velocity to true vel
            rVel = v_f-v_s
            #pressure contrib
            rho_s=partRho[i]
            rho_f=partRho[listFluidNb]
            P_s=pressure(rho_s,B,rhoS,gamma)
            P_f=pressure(rho_f,B,rhoF,gamma)
            FFluidSolid = FFluidSolidContrib(P_f, P_s, mu, rho_f, rho_s,dwdr,rVel,rPos,m,ms)
            # We sum the contrib for all fluid particles
            forces[i,:] = np.sum(F,0)
    return forces

@njit
def IntegrateCenterOfMassMovement(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,mu,OG, V_OG,dt):
    A_OG = grav + np.sum(computeForcesFluidSolid(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,mu),0)
    V_OG += A_OG*dt
    OG += V_OG*dt
    
