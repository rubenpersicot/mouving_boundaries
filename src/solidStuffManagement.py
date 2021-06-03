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
from src.spyh import *



   
#@njit
def computeForcesFluidSolid(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,ms,B,rhoF,rhoS,gamma,grav,solidAcc,mu,d=2):
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
            - ms : solid particle mass
            - B : state constant
            - rhoF : reference density
            - rhoS : solid reference density
            - gamma : polytropic gas constant
            - grav : gravitational acceleration
            - solidAcc : the acceleration of center of mass of the solid
            - mu : viscosity
            - d : dimension
        return :
            - forces : table of the forces on each solid particle
    '''
    forces = np.zeros_like(partVel)
    drhodt = np.zeros_like(partRho)
    nPart = len(partMOBILESOLID)
    for i in range(nPart):
        if partMOBILESOLID[i]:
            spid_i = int(partSPID[i])
            #list neib
            listnb = listNeibSpace[spid_i,:]
            listnb = listnb[listnb>-1]
            listnb = listnb[listnb!=i] #no self contribution
            #---------------------------------------------------
            #removing non fluid particle from the neighbours list
            listnb = listnb[partMOBILESOLID[listnb]==False]
            #---------------------------------------------------
            #Position, norm  and er
            rPos = partPos[listnb][:]-partPos[i,:]
            rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
            q=rNorm/h
            dwdr = Fw(q,aW,h)
            w_ij = wend(q,aW,h)
            er = np.zeros_like(rPos)
            er[:,0] = rPos[:,0]/rNorm
            er[:,1] = rPos[:,1]/rNorm
            #velocity
            v_s = partVel[i,:]
            v_f = partVel[listnb][:]
            rVel = v_f-v_s
            #pressure contrib
            rho_s=partRho[i]
            rho_f=partRho[listnb]
            vol_f=m/rho_f
            P_f=pressure(rho_f,B,rhoF,gamma)
            #solid particle pressure calculated according to  Cf Adami, Hu & Adams, 2012
            P_s = pressureInterpolationContrib(rho_f, P_f, vol_f,rPos,w_ij,grav,solidAcc)
            P_s[P_s<0] = 0            
            FFluidSolid = FFluidSolidContrib(P_f, P_s, mu, rho_f, rho_s,dwdr,rVel,rPos,m,ms)
            # We sum the contrib for all fluid particles
            forces[i,:] = np.sum(FFluidSolid,0)
    return forces

#@njit
def IntegrateCenterOfMassMovement(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,ms,B,rhoF,rhoS,gamma,grav,mu,OG, V_OG,A_OG,part,nSolid,dt):
    '''
        Computes the movement of the center of mass
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
            - ms : solid particle mass
            - B : state constant
            - rhoF : reference density
            - gamma : polytropic gas constant
            - grav : gravitational acceleration
            - OG : the current position of the center of mass
            - V_OG : the current velocity of the center of mass
            - A_OG : the current acceleration of the center of mass
            - mu : viscosity
            - d : dimension
        return :
            - doG : the delta of displacement of the center of mass
            - dV_OG : the delta of velocity of the center of mass
            - A_OG : the new acceleration of the center of mass
    '''
    nPart = len(partMOBILESOLID)
    centerOfMassPosX, centerOfMassPosY = 0.0,0.0
    #compute the center of mass position
    for i in range(nPart):
        if partMOBILESOLID[i]:
            centerOfMassPosX += partPos[i,0]
            centerOfMassPosY += partPos[i,1]
    centerOfMassPos = 1/nSolid*np.array([centerOfMassPosX, centerOfMassPosY])
    #if we are above the free surface, no forces are computed
    if (centerOfMassPos[1]-0.2)>1:
        F=np.array([0.0,0.0])
    else :
        F = np.sum(computeForcesFluidSolid(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,aW,h,m,ms,B,rhoF,       rhoS,gamma,grav,A_OG,mu),0)
    print("Force fluid -> solid :")
    print(F)
    A_OG = grav + F/(ms*nSolid)
    dV_OG = A_OG*dt
    dOG = V_OG*dt
    return dOG, dV_OG, A_OG

@njit
def MoveSolidParticles(partMOBILESOLID, partPos, partVel, dOG, dV_OG):
    '''
    computes the global movement of the solid which is considered rigid
    inputs :
            - partMOBILESOLID : True if a MOBILESOLID particle
            - partPos : particle position
            - partVel : particle velocity
            - doG : the delta of displacement of the center of mass
            - dV_OG : the delta of velocity of the center of mass 
    outputs : 
            -partPos : the updated table of position
            -partVel : the updated table of velocity
    '''
    nPart = len(partMOBILESOLID)
    for i in range(nPart):
        if partMOBILESOLID[i]:
            partPos[i,0]=partPos[i,0]+dOG[0]
            partPos[i,1]=partPos[i,1]+dOG[1]
            partVel[i,0]=partVel[i,0]+dV_OG[0]
            partVel[i,1]=partVel[i,1]+dV_OG[1]
    return partPos, partVel 

#@njit
def interpolateMobileSolidBoundary(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,solidVel,solidAcc,shepardMin = 10**(-6),d=2):
    '''
    interpolate the pressure and velocity on the mobile solid
    input : 
        - partMOBILESOLID : table of True,False showing which particle is a MOBILESOLID
        - partSPID : table of particles SPIDs
        - partPos : table of particles positions
        - partVel : table of particles velocities
        - partRho : table of particles density
        - partSpace : table of particle per spaces (-1 are no values)
        - neibSpace : table of spaces neibs
        - aW,d,h : parameters for the wendland
        - m : particle mass
        - B, rhoF, gamma : state equation parameters
        - grav : gravital field acceleration
        - solidVel : velocity of the solid particles --> constant as the solid is considered rigid
        - solidAcc : acceleration of the solid particles
        - shepardMin : threshold for the shepard
    output : 
        - partRho : updated table of density with interpolated pressure 
        - partVel : new mobile solid velocity
    '''
    nPart = len(partMOBILESOLID)
    for i in range(nPart):
            if partMOBILESOLID[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                # keep only the fluid particles 
                listnb = listnb[partMOBILESOLID[listnb]==False]
                rPos = partPos[i,:]-partPos[listnb][:]
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                w_ij = wend(q,aW,h)
                rho_j=partRho[listnb]
                P_j= pressure(rho_j,B,rhoF,gamma)
                rho_j[rho_j<rhoF] = rhoF
                vol_j = m/rho_j
                rho_j= partRho[listnb]
                PressInt = pressureInterpolationContrib(rho_j, P_j, vol_j,rPos,w_ij,grav,solidAcc)
                PressInt[PressInt<0] = 0
                shepard = shepardContrib(vol_j,w_ij)
                shepard = max(np.sum(shepard,0), shepardMin)
                #VELOCITY INTERPOLATION FOR MOBILE BOUNDARIES according to S. Adami's work
                VTildeInt_x = solidVel[0]*vol_j*w_ij 
                VTildeInt_y = solidVel[1]*vol_j*w_ij
                partVel[i,0] = 2*solidVel[0] - np.sum(VTildeInt_x,0)/shepard
                partVel[i,1] = 2*solidVel[1] - np.sum(VTildeInt_y,0)/shepard
                pres = np.sum(PressInt,0)/shepard
                partRho[i] = density(pres,B,rhoF,gamma)
    return partRho,partVel