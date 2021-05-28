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
                        aW,h,m,ms,B,rhoF,gamma,grav,mu,d=2):
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
            listnb = listnb[partMOBILESOLID[listnb]==False]
            #---------------------------------------------------
            #Position, norm  and er
            rPos = partPos[listnb][:]-partPos[i,:]
            rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
            q=rNorm/h
            dwdr = Fw(q,aW,h)
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
            P_s=pressure(rho_s,B,rhoS,gamma)
            P_f=pressure(rho_f,B,rhoF,gamma)
            FFluidSolid = FFluidSolidContrib(P_f, P_s, mu, rho_f, rho_s,dwdr,rVel,rPos,m,ms)
            # We sum the contrib for all fluid particles
            forces[i,:] = np.sum(F,0)
    return forces

@njit
def IntegrateCenterOfMassMovement(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,ms,B,rhoF,gamma,grav,mu,OG, V_OG,dt):
    A_OG = grav + np.sum(computeForcesFluidSolid(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,ms,B,rhoF,gamma,grav,mu),0)/ms
    V_OG += A_OG*dt
    OG += V_OG*dt
    return OG, V_OG, A_OG


@njit
def interpolateMobileSolidBoundary(partMOBILESOLID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,solidVel,shepardMin = 10**(-6),d=2):
    '''
    interpolate the pressure and velocity at the walls
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
        - SolidVel : velocity of the solid particles --> constant as the solid is considered rigid
        - shepardMin : threshold for the shepard
    output : 
        - partRho : updated table of density with interpolated pressure 
        - partVel : fictious boundary velocity (not true velocity)
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
                #
                rho_j=partRho[listnb]
                P_j= pressure(rho_j,B,rhoF,gamma)
                rho_j[rho_j<rhoF] = rhoF
                vol_j = m/rho_j
                rho_j= partRho[listnb]
                PressInt = pressureInterpolationContrib(rho_j, P_j, vol_j,rPos,w_ij,grav)
                PressInt[PressInt<0] = 0
                shepard = shepardContrib(vol_j,w_ij)
                shepard = max(np.sum(shepard,0), shepardMin)
                #VELOCITY INTERPOLATION FOR MOBILE BOUNDARIES according to S. Adami's work
                VTildeInt_x = SolidVel[0]*vol_j*w_ij 
                VTildeInt_y = SolidVel[1]*vol_j*w_ij
                partVel[i,0] = 2*SolidVel[0] - np.sum(VTildeInt_x,0)/shepard
                partVel[i,1] = 2*SolidVel[1] - np.sum(VTildeInt_y,0)/shepard
                pres = np.sum(PressInt,0)/shepard
                partRho[i] = density(pres,B,rhoF,gamma)
    return partRho,partVel