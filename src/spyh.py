###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
#Authors :  R. Carmigniani & D. Violeau
#Version : SPyH
#Contact : remi.carmigniani@enpc.fr
###########################################################################
# Kernel Wendland
import numpy as np
from numba import njit
from src.sphvar import *
from src.state import *
from src.contrib import *

#Wendland kernel
@njit
def wend(q,aW,h,d=2):
    '''
    Wendland kernel
    input :
        - q :  np.array of r/h values
        - aW : coefficient of the wendland kernel
        - h : smoothing length
        - d : dimension
    return :
        - w
    '''
    res =aW/h**d*(1-q*0.5)**4*(1+2*q)
    res[q>2] = 0
    return res

@njit
def Fw(q,aW,h,d=2):
    '''
    Gradient of the kernel 
     input :
        - q :  np.array of r/h values
        - aW : coefficient of the wendland kernel
        - h : smoothing length
        - d : dimension
    return :
        - Fw
    '''
    res =aW/h**(d+1)*5/8*q*(-2+q)**3
    res[q>2] = 0
    return res


def init_particles():
    '''
        return an empty table for the particle
        nparameters is a global parameter in sphvar.py
    '''
    return  np.empty((0,nparameters),float)

def addBox(part,domain,TYPE,dr,rhoF):
    '''
    add a box of particle of type TYPE
    input :
        - part : the current particle table
        - domain: accepted format 
            [lx,ly] : the size of the domain to fill with particles with origin 0,0
            [x0,y0,lx,ly] : the size of the domain to fill with particles with origin x0,y0
        - TYPE : the type of particle to create (see src/config/sphvar.py for the TYPEs 
        - dr : particle size
        - rhoF : particle density
    return : 
        - part : updated particle table
    '''
    if len(domain)==2:
        x0 = 0
        y0 = 0
        lx = domain[0]
        ly = domain[1]
    elif len(domain)==4:
        x0 = domain[0]
        y0 = domain[1]
        lx = domain[2]
        ly = domain[3]
    else:
        print('ERROR in create Box')
    x = x0+dr*0.5
    while x<=x0+lx:
        y=y0+dr*0.5
        while y<=y0+ly:
            partnew = np.zeros((1,nparameters))
            partnew[0][POS] = [x,y]
            partnew[0][RHO] = rhoF
            partnew[0][INFO] = TYPE
            part = np.append(part,partnew,axis=0)
            y=y+dr
        x=x+dr
    return part

def init_spaces(xOrigin,yOrigin,xSize,ySize,lspace,dr,vecPer=[0,0]):
    '''
    list of spaces
    input :
        - xOrigin, yOrigin,xSize,ySize : universe dimension
        - lspace : space size
        - dr : particle size
        - vecPer : periodic vector [valid for 1 direction periodicity so far]
    return :
        -centers : contains the coordinates of the center of the spaces
        -spaceNeibs : contains the list of the neighbour spaces
        -partInSpace : the list of Ids of the particle in the space
        -listNeibSpace : table of neibs particles per spaces
    '''
    posSpace = np.empty((0,2),float)
    spaceCount = 1
    #Update xmin xmax ymin ymax
    #print('Update xmin,xmax,ymin,ymax')
    xmin =xOrigin
    ymin =yOrigin
    xmax =xOrigin+xSize
    ymax =yOrigin+ySize
    #
    x0 = xOrigin+lspace/2
    y0 = yOrigin+lspace/2
    while (x0<xmax+lspace/4):
        y0 = yOrigin+lspace/2
        while (y0<ymax+lspace/4):
            posSpace = np.append(posSpace,[[x0,y0]],axis=0)
            y0+=lspace
            spaceCount+=1
        x0+=lspace
    nSpace = len(posSpace)
    #Create the empty tables
    neibSpace = np.ones((nSpace,maxSpaceNeibs),int)*-1
    partSpace = np.ones((nSpace,maxPartInSpace),int)*-1
    listNeibSpace = np.ones((nSpace,maxSpaceNeibs*maxPartInSpace),int)*-1
    countNeib = np.zeros(nSpace,int)
    for i in range(nSpace):
        neibSpace[i][countNeib[i]] = i
        countNeib[i] +=1
        j=i+1
        while (j<nSpace):
            # WARNING WORKS ONLY FOR LSPACE >=2*h
            if (np.linalg.norm(posSpace[i]-posSpace[j])) < 1.5*lspace:
                neibSpace[i][countNeib[i]] = j
                neibSpace[j][countNeib[j]] = i
                countNeib[i] +=1
                countNeib[j] +=1
            elif (np.linalg.norm(posSpace[i]-(posSpace[j]+vecPer)))<1.5*lspace:
                neibSpace[i][countNeib[i]] = j
                neibSpace[j][countNeib[j]] = i
                countNeib[i] +=1
                countNeib[j] +=1
            elif (np.linalg.norm(posSpace[i]-(posSpace[j]-vecPer)))<1.5*lspace:
                neibSpace[i][countNeib[i]] = j
                neibSpace[j][countNeib[j]] = i
                countNeib[i] +=1
                countNeib[j] +=1
            j=j+1
            
            
    return posSpace,neibSpace,partSpace,listNeibSpace

@njit
def sortPartTable(posTab,xOrigin,yOrigin,xSize,ySize,lspace,dr,maxpos0,maxpos1):
    '''
    sort the particles : this function return the space Id position of the particles
    the -1000 is set for particles outside of the physical domain
    input :
        - posTab : table of particle positions
        - xOrigin, yOrigin,xSize,ySize : universe dimension
        - lspace : space size
        - dr : particle size
        - maxpos0, maxpos1 : maximum position of the spaces [could be improved]
    return :
        -spID : table of the particle space ID positions
    '''
    nPart = len(posTab)
    spID = np.zeros(nPart)
    spID = (np.floor((posTab[:,1]-yOrigin)/lspace))\
        +(np.floor((posTab[:,0]-xOrigin)/lspace))\
        *np.floor((maxpos1-yOrigin+0.5*dr)/lspace)
    # FLAG THE PARTICLES TO DELETE :
    spID[posTab[:,1]<yOrigin] = -1000
    spID[posTab[:,1]>yOrigin+ySize]= -1000
    spID[posTab[:,0]<xOrigin] = -1000
    spID[posTab[:,0]>xOrigin+xSize] = -1000
    return spID

def sortPart(part,posSpace,partSpace,xOrigin,yOrigin,xSize,ySize,lspace,dr):
    '''
    sort the particles and give the particle indices in each spaces
    input :
        - part : particles table 
        - posSpace : centers of the spaces
        - partSpace : table of the particules in the spaces
        - xOrigin, yOrigin, xSize,ySize : computation world informations
        - lspace :  size of spaces
        - dr : particle size
    output :
        - part :  sorted table of particles
        - partSpace : table of particles in a space
    '''
    nPart = len(part)
    posTab = part[:,POS]
    nSpace = len(posSpace)
    maxpos0 = max(posSpace[:,0])+lspace/2
    maxpos1 = max(posSpace[:,1])+lspace/2
    '''
    sort the part table
    '''
    spID = sortPartTable(posTab,xOrigin,yOrigin,xSize,ySize,lspace,dr,maxpos0,maxpos1)
    part[:,SPID] = spID
    #Sort the Particles 
    I = np.argsort(spID)
    part = part[I,:]
    #Delete the particle out of the computational domain: 
    if part[0,SPID]<0:#There exists particles outside
        ids = np.nonzero(part[:,SPID] == -1000)
        ids = ids[0]
        print(repr(len(ids))+' particles are deleted because outside computational domain')
        part = part[ids[-1]+1:][:]
    #Update the Space informations :
    partSpace, partCount = updatePartSpace(partSpace,part[:,SPID],maxPartInSpace)
    if partCount>maxPartInSpace:
        print('ATTENTION : too many neibs in a space... increase maxPartInSpace in sphvar.py')
    return part,partSpace

@njit
def updatePartSpace(partSpace,partSPID,maxPartInSpace):
    '''
    update the table of particles in a space
    input :
        - partSpace : old table of the particles in a space
        - partSPID : new table of the particle space ID. 
        - maxPartInSpace : maximum number of particle in a space
    output :
        - partSpace : updated table of the particles in a space
        - partCountMax : maximum of the particle in a space. Should be less than maxPartInSPace
    '''
    nSpace = len(partSpace)
    partCountMax = 0
    for i in range(nSpace):
        partSpace[i] = -1
        partToAdd = np.nonzero(partSPID == i)[0] 
        partCount = len(partToAdd)
        partCountMax = max(partCountMax,partCount)
        for j in range(min(partCount
                           ,maxPartInSpace)):
            partSpace[i,j] = partToAdd[j]
    return partSpace, partCountMax

@njit
def getListNeib(partSpace,neibSpace,listNeibSpace):
    '''
    return the list of the particles influencing a particle in a space
    input : 
        - partSpace :  table of particle per spaces (-1 are no values)
        - neibSpace :  table of spaces neibs
        - listNeibSpace : old list of the particles influencing a particle in a space spid
    return : 
        - listNeibSpace : new list of the particles influencing a particle in a space spid
    '''
    nSpace = len(partSpace)
    for spid in range(nSpace):
        listSpaces = neibSpace[spid][neibSpace[spid]>-1]
        #TODO COMPLETE HERE
        listpart = partSpace[listSpaces].flatten()
        listpart = listpart[listpart>-1]
        nPart = len(listpart)
        listNeibSpace[spid,:]=-1
        for i in range(nPart):
            listNeibSpace[spid,i] = listpart[i]
        # END
    return listNeibSpace


@njit
def interpolateBoundary(partBOUND,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,shepardMin = 10**(-6),d=2):
    '''
    interpolate the pressure and velocity at the walls
    input : 
        - partBOUND : table of True,False showing which particle is a Boundary
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
        - shepardMin : threshold for the shepard
    output : 
        - partRho : updated table of density with interpolated pressure 
        - partVel : fictious boundary velocity (not true velocity)
    '''
    nPart = len(partBOUND)
    for i in range(nPart):
            if partBOUND[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                # keep only the fluid particles 
                listnb = listnb[partBOUND[listnb]==False]
                rPos = partPos[i,:]-partPos[listnb][:]
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                w_ij = wend(q,aW,h)
                #
                rho_j=partRho[listnb]
                vel_j = partVel[listnb][:]
                P_j= pressure(rho_j,B,rhoF,gamma)
                rho_j[rho_j<rhoF] = rhoF
                vol_j = m/rho_j
                rho_j= partRho[listnb]
                PressInt = pressureInterpolationContrib(rho_j, P_j, vol_j,rPos,w_ij,grav)
                PressInt[PressInt<0] = 0
                shepard = shepardContrib(vol_j,w_ij)
                shepard = max(np.sum(shepard,0), shepardMin)
                #COMPLETE HERE
                VTildeInt_x = -vel_j[:,0]*vol_j*w_ij
                VTildeInt_y = -vel_j[:,1]*vol_j*w_ij
                partVel[i,0] = np.sum(VTildeInt_x,0)/shepard
                partVel[i,1] = np.sum(VTildeInt_y,0)/shepard
                #END
                pres = np.sum(PressInt,0)/shepard
                partRho[i] = density(pres,B,rhoF,gamma)
    return partRho,partVel

@njit
def interpolateBoundaryPeriodicX(partBOUND,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,xper,shepardMin = 10**(-6),d=2):
    '''
    interpolate the pressure and velocity at the walls
    input : 
        - partBOUND : table of True,False showing which particle is a Boundary
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
        - xper : periodicity length in x
        - shepardMin : threshold for the shepard
    output : 
        - partRho : updated table of density with interpolated pressure
        - partVel : updated table of velocity using Adami et al. 
    '''
    nPart = len(partBOUND)
    for i in range(nPart):
            if partBOUND[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                # keep only the fluid particles 
                listnb = listnb[partBOUND[listnb]==False]
                rPos = partPos[i,:]-partPos[listnb][:]
                rPos[:,0] += (rPos[:,0]>xper/2)*(-1*xper)+(rPos[:,0]<-xper/2)*xper
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                w_ij = wend(q,aW,h)
                #
                rho_j=partRho[listnb]
                vel_j = partVel[listnb][:]
                P_j= pressure(rho_j,B,rhoF,gamma)
                rho_j[rho_j<rhoF] = rhoF
                vol_j = m/rho_j
                rho_j= partRho[listnb]
                PressInt = pressureInterpolationContrib(rho_j, P_j, vol_j,rPos,w_ij,grav)
                PressInt[PressInt<0] = 0
                shepard = shepardContrib(vol_j,w_ij)
                shepard = max(np.sum(shepard,0), shepardMin)
                #COMPLETE HERE
                VTildeInt_x = -vel_j[:,0]*vol_j*w_ij
                VTildeInt_y = -vel_j[:,1]*vol_j*w_ij
                partVel[i,0] = np.sum(VTildeInt_x,0)/shepard
                partVel[i,1] = np.sum(VTildeInt_y,0)/shepard
                #END
                pres = np.sum(PressInt,0)/shepard
                partRho[i] = density(pres,B,rhoF,gamma)
    return partRho,partVel

def initMobileBoundVelocity(partMOBILEBOUND, partVel, U):
    '''
    Initialisation of mobile bound particle velocity:
        - partMOBILEBOUND : table of True,False showing which particle is a mobile Boundary
        - partVel : table of particles velocities
        - U : upper boundary velocity
    '''
    nPart = len(partMOBILEBOUND)
    for i in range(nPart):
        if partMOBILEBOUND[i]:
            partVel[i][0] = U
    return partVel

@njit
def interpolateMobileBoundaryPeriodicX(partMOBILEBOUND,partSPID,partPos,partVel, wallVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,xper,shepardMin = 10**(-6),d=2):
    '''
    interpolate the pressure and velocity at the walls
    input : 
        - partMOBILEBOUND : table of True,False showing which particle is a mobile Boundary
        - partSPID : table of particles SPIDs
        - partPos : table of particles positions
        - partVel : table of particles velocities
        - wallVel : prescribed wall velocity
        - partRho : table of particles density
        - partSpace : table of particle per spaces (-1 are no values)
        - neibSpace : table of spaces neibs
        - aW,d,h : parameters for the wendland
        - m : particle mass
        - B, rhoF, gamma : state equation parameters
        - grav : gravital field acceleration
        - xper : periodicity length in x
        - shepardMin : threshold for the shepard
    output : 
        - partRho : updated table of density with interpolated pressure
        - partVel : updated table of velocity using Adami et al. 
    '''
    nPart = len(partMOBILEBOUND)
    for i in range(nPart):
            if partMOBILEBOUND[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                # keep only the fluid particles 
                listnb = listnb[partMOBILEBOUND[listnb]==False]
                rPos = partPos[i,:]-partPos[listnb][:]
                rPos[:,0] += (rPos[:,0]>xper/2)*(-1*xper)+(rPos[:,0]<-xper/2)*xper
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                w_ij = wend(q,aW,h)
                #
                rho_j=partRho[listnb]
                vel_j = partVel[listnb][:]
                P_j= pressure(rho_j,B,rhoF,gamma)
                rho_j[rho_j<rhoF] = rhoF
                vol_j = m/rho_j
                rho_j= partRho[listnb]
                PressInt = pressureInterpolationContrib(rho_j, P_j, vol_j,rPos,w_ij,grav)
                PressInt[PressInt<0] = 0
                shepard = shepardContrib(vol_j,w_ij)
                shepard = max(np.sum(shepard,0), shepardMin)
                #VELOCITY INTERPOLATION FOR MOBILE BOUNDARIES according to S. Adami's work
                VTildeInt_x = wallVel[0]*vol_j*w_ij 
                VTildeInt_y = wallVel[1]*vol_j*w_ij
                partVel[i,0] = 2*wallVel[0] - np.sum(VTildeInt_x,0)/shepard
                partVel[i,1] = 2*wallVel[1] - np.sum(VTildeInt_y,0)/shepard
                pres = np.sum(PressInt,0)/shepard
                partRho[i] = density(pres,B,rhoF,gamma)
    return partRho, partVel


@njit
def CFLConditions(partVel,h,c0,grav,rhoF,mu,CFL=0.1):
        '''
        return a acceptable time step
        dt1 : based on the speed of sound
        dt2 : based on the gravity
        dt3 : based on the viscosity
        output : the minimum of the 3
        '''
        vmax = np.max((partVel[:,0]*partVel[:,0]+partVel[:,1]*partVel[:,1])**0.5)
        dt1 = CFL*h/(c0+vmax)
        #dt2 = CFL*(h/np.linalg.norm(grav))**.5
        dt3 = CFL*0.5*rhoF*h**2/mu
        return min(dt1,dt3)
    
@njit   
def computeForcesART(partFLUID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                    aW,h,m,B,rhoF,gamma,grav,alpha,eps,dr,d=2):
        '''
        Compute the forces using artificial viscous forces
        and the RHS for the continuity equation
        input :
            - partFLUID : True if a FLUID particle
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
            - alpha, eps, dr : for the artificial viscosity
            - d : dimension
        return :
            - forces : table of the particle forces
            - drhodt : table of the density variation
        '''
        forces = np.zeros_like(partVel)
        drhodt = np.zeros_like(partRho)
        nPart = len(partFLUID)
        for i in range(nPart):
            if partFLUID[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                #Position, norm  and er
                rPos = partPos[i,:]-partPos[listnb][:]
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                dwdr = Fw(q,aW,h)
                er = np.zeros_like(rPos)
                er[:,0] = rPos[:,0]/rNorm
                er[:,1] = rPos[:,1]/rNorm
                #velocity
                v_i = partVel[i,:]
                #Continuity Velocity :  the velocity is the true wall velocity 
                #for now only static walls are considered :
                v_j = partVel[listnb][:]
                v_j[partFLUID[listnb]==False][:,0]= 0 #set wall velocity to true vel
                v_j[partFLUID[listnb]==False][:,1]= 0 #set wall velocity to true vel
                rVelCont = v_i-v_j
                #Viscosity velocity
                # use interpolated velocity (see next TD)
                rVelViscous = v_i-partVel[listnb][:]
                #pressure contrib
                rho_i=partRho[i]
                rho_j=partRho[listnb]
                P_i=pressure(rho_i,B,rhoF,gamma)
                P_j=pressure(rho_j,B,rhoF,gamma)
                F_Pres = pressureGradContrib(rho_i,rho_j,P_i,P_j,dwdr,er,m)
                #viscous contribution
                c_i = soundSpeed(rho_i,B,rhoF,gamma)
                c_j = soundSpeed(rho_j,B,rhoF,gamma)
                mu_art = rho_i*h*alpha*(rho_j*(c_i+c_j)/(rho_i+rho_j))
                mu_art[partFLUID[listnb]==False]=0 #no wall contrib
                F_visc = artViscContrib(mu_art,rho_i, rho_j,dwdr,rVelViscous,rPos,m,dr,eps)
                # Add the forces contrib
                forces[i,:] = np.sum(F_Pres,0)+np.sum(F_visc,0)+grav
                #continuity contribution
                drhodt[i] = np.sum(velocityDivContrib(rVelCont,rPos,dwdr,er,m),0)
        return forces,drhodt

    
@njit   
def computeForcesARTPeriodicX(partFLUID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                    aW,h,m,B,rhoF,gamma,grav,alpha,eps,dr,xper,d=2):
        '''
        Compute the forces using artificial viscous forces
        and the RHS for the continuity equation
        PeriodicX is for periodic in the x direction
        input :
            - partFLUID : True if a FLUID particle
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
            - alpha, eps, dr : for the artificial viscosity
            - d : dimension
        return :
            - forces : table of the particle forces
            - drhodt : table of the density variation
        '''
        forces = np.zeros_like(partVel)
        drhodt = np.zeros_like(partRho)
        nPart = len(partFLUID)
        for i in range(nPart):
            if partFLUID[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                #Position, norm  and er
                rPos = partPos[i,:]-partPos[listnb][:]
                rPos[:,0] += (rPos[:,0]>xper/2)*(-1*xper)+(rPos[:,0]<-xper/2)*xper
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                dwdr = Fw(q,aW,h)
                er = np.zeros_like(rPos)
                er[:,0] = rPos[:,0]/rNorm
                er[:,1] = rPos[:,1]/rNorm
                #velocity
                v_i = partVel[i,:]
                #Continuity Velocity :  the velocity is the true wall velocity 
                #for now only static walls are considered :
                v_j = partVel[listnb][:]
                v_j[partFLUID[listnb]==False][:,0]= 0 #set wall velocity to true vel
                v_j[partFLUID[listnb]==False][:,1]= 0 #set wall velocity to true vel
                rVelCont = v_i-v_j
                #Viscosity velocity
                # use interpolated velocity (see next TD)
                rVelViscous = v_i-partVel[listnb][:]
                #pressure contrib
                rho_i=partRho[i]
                rho_j=partRho[listnb]
                P_i=pressure(rho_i,B,rhoF,gamma)
                P_j=pressure(rho_j,B,rhoF,gamma)
                F_Pres = pressureGradContrib(rho_i,rho_j,P_i,P_j,dwdr,er,m)
                #viscous contribution
                c_i = soundSpeed(rho_i,B,rhoF,gamma)
                c_j = soundSpeed(rho_j,B,rhoF,gamma)
                mu_art = rho_i*h*alpha*(rho_j*(c_i+c_j)/(rho_i+rho_j))
                mu_art[partFLUID[listnb]==False]=0 #no wall contrib
                F_visc = artViscContrib(mu_art,rho_i, rho_j,dwdr,rVelViscous,rPos,m,dr,eps)
                # Add the forces contrib
                forces[i,:] = np.sum(F_Pres,0)+np.sum(F_visc,0)+grav
                #continuity contribution
                drhodt[i] = np.sum(velocityDivContrib(rVelCont,rPos,dwdr,er,m),0)
        return forces,drhodt

    
@njit
def integrationStep(partFLUID,partPos,partVel,partRho,partFORCES,partDRHODT,dt):
    '''
    Euler explicit integration step with simpletic scheme 
    input : 
            - partFLUID : True if a FLUID particle
            - partPos : particle position
            - partVel : particle velocity
            - partRho : particle density
            - partFORCES : particle forces
            - partDRHODT : particle density variation
            - dt : particle time step
    return :
            - partPos : updated particle position
            - partVel : updated particle velocity
            - partRho : updated particle density
    '''
    nPart = len(partPos)
    for i in range(nPart):
        if partFLUID[i]:
            #TODO COMPLETE HERE
            partVel[i,:] += dt*partFORCES[i,:]
            partPos[i,:] += dt*partVel[i,:]
            partRho[i] += dt*partDRHODT[i]
            #END
    return partPos,partVel,partRho

@njit
def integrationStepPeriodicX(partFLUID,partPos,partVel,partRho,partFORCES,partDRHODT,dt,xper):
    '''
    Euler explicit integration step with simpletic scheme
    input : 
            - partFLUID : True if a FLUID particle
            - partPos : particle position
            - partVel : particle velocity
            - partRho : particle density
            - partFORCES : particle forces
            - partDRHODT : particle density variation
            - dt : particle time step
    return :
            - partPos : updated particle position
            - partVel : updated particle velocity
            - partRho : updated particle density
    '''
    nPart = len(partPos)
    for i in range(nPart):
        if partFLUID[i]:
            #TODO COMPLETE HERE
            partVel[i,:] += dt*partFORCES[i,:]
            partPos[i,:] += dt*partVel[i,:]
            partRho[i] += dt*partDRHODT[i]
            #END
    #periodicity        
    partPos[:,0] = np.mod(partPos[:,0],xper)
    return partPos,partVel,partRho
    
@njit
def integrationStepPeriodicX_Moving_Bound(partMOBILEBOUND,partPos,partVel,partRho,partFORCES,partDRHODT,dt,xper):
    '''
    Euler explicit integration step for mobile bound particle
    input : 
            - partMOBILEBOUND : True if a MOBILEBOUND particle
            - partPos : particle position
            - partVel : particle velocity
            - partRho : particle density
            - partFORCES : particle forces
            - partDRHODT : particle density variation
            - dt : particle time step
    return :
            - partPos : updated particle position
            - partVel : updated particle velocity
            - partRho : updated particle density
    '''
    nPart = len(partPos)
    for i in range(nPart):
        if partMOBILEBOUND[i]:
            #TODO COMPLETE HERE
            partPos[i,:] += dt*partVel[i,:]
            partRho[i] += dt*partDRHODT[i]
            #END
    #periodicity        
    partPos[:,0] = np.mod(partPos[:,0],xper)
    return partPos, partRho
    
@njit 
def checkDensity(partRho,rhoMin,rhoMax):
    '''
    check density and print a warning
    '''
    if (partRho[partRho<rhoMin]).size>0:
            partRho[partRho<rhoMin]=rhoMin
            print('WARNING : density dropped weirdly for some particles')
    if (partRho[partRho>rhoMax]).size>0:
            partRho[partRho>rhoMax]=rhoMax
            print('WARNING : density increased weirdly for some particles')
    return partRho



@njit   
def computeForcesMorrisPeriodicX(partFLUID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                    aW,h,m,B,rhoF,gamma,grav,mu,xper,d=2):
        '''
        Compute the forces using Morris viscous forces
        and the RHS for the continuity equation
        Periodic X :  periodicity of length xper
        input :
            - partFLUID : True if a FLUID particle
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
            - drhodt : table of the density variation
        '''
        forces = np.zeros_like(partVel)
        drhodt = np.zeros_like(partRho)
        nPart = len(partFLUID)
        #TODO : COMPLETE HERE
        for i in range(nPart):
            if partFLUID[i]:
                spid_i = int(partSPID[i])
                #list neib
                listnb = listNeibSpace[spid_i,:]
                listnb = listnb[listnb>-1]
                listnb = listnb[listnb!=i] #no self contribution
                #Position, norm  and er
                rPos = partPos[i,:]-partPos[listnb][:]
                rPos[:,0] += (rPos[:,0]>xper/2)*(-1*xper)+(rPos[:,0]<-xper/2)*xper
                rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
                q=rNorm/h
                dwdr = Fw(q,aW,h)
                er = np.zeros_like(rPos)
                er[:,0] = rPos[:,0]/rNorm
                er[:,1] = rPos[:,1]/rNorm
                #velocity
                v_i = partVel[i,:]
                #Continuity Velocity :  the velocity is the true wall velocity 
                v_j = partVel[listnb][:]
                #v_j[partFLUID[listnb]==False][:,0]= 0 #set wall velocity to true vel
                #v_j[partFLUID[listnb]==False][:,1]= 0 #set wall velocity to true vel
                rVelCont = v_i-v_j
                #Viscosity velocity
                # use interpolated velocity (see next TD)
                rVelViscous = v_i-partVel[listnb][:]
                #pressure contrib
                rho_i=partRho[i]
                rho_j=partRho[listnb]
                P_i=pressure(rho_i,B,rhoF,gamma)
                P_j=pressure(rho_j,B,rhoF,gamma)
                F_Pres = pressureGradContrib(rho_i,rho_j,P_i,P_j,dwdr,er,m)
                #viscous contribution
                F_visc = MorrisViscContrib(mu,rho_i, rho_j,dwdr,rVelViscous,rPos,m)
                # Add the forces contrib
                forces[i,:] = np.sum(F_Pres,0)+np.sum(F_visc,0)+grav
                #continuity contribution
                drhodt[i] = np.sum(velocityDivContrib(rVelCont,rPos,dwdr,er,m),0)
        #END
        return forces,drhodt

@njit   
def computeForcesMorris(partFLUID,partSPID,partPos,partVel,partRho,listNeibSpace,\
                        aW,h,m,B,rhoF,gamma,grav,mu,d=2):
    '''
        Compute the forces using Morris viscous forces
        and the RHS for the continuity equation
        input :
            - partFLUID : True if a FLUID particle
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
            - drhodt : table of the density variation
    '''
    forces = np.zeros_like(partVel)
    drhodt = np.zeros_like(partRho)
    nPart = len(partFLUID)
    for i in range(nPart):
        if partFLUID[i]:
            spid_i = int(partSPID[i])
            #list neib
            listnb = listNeibSpace[spid_i,:]
            listnb = listnb[listnb>-1]
            listnb = listnb[listnb!=i] #no self contribution
            #Position, norm  and er
            rPos = partPos[i,:]-partPos[listnb][:]
            rNorm = (rPos[:,0]*rPos[:,0]+rPos[:,1]*rPos[:,1])**.5
            q=rNorm/h
            dwdr = Fw(q,aW,h)
            er = np.zeros_like(rPos)
            er[:,0] = rPos[:,0]/rNorm
            er[:,1] = rPos[:,1]/rNorm
            #velocity
            v_i = partVel[i,:]
            #Continuity Velocity :  the velocity is the true wall velocity 
            v_j = partVel[listnb][:]
            #v_j[partFLUID[listnb]==False][:,0]= 0 #set wall velocity to true vel
            #v_j[partFLUID[listnb]==False][:,1]= 0 #set wall velocity to true vel
            rVelCont = v_i-v_j
            #Viscosity velocity
            # use interpolated velocity (see next TD)
            rVelViscous = v_i-partVel[listnb][:]
            #pressure contrib
            rho_i=partRho[i]
            rho_j=partRho[listnb]
            P_i=pressure(rho_i,B,rhoF,gamma)
            P_j=pressure(rho_j,B,rhoF,gamma)
            F_Pres = pressureGradContrib(rho_i,rho_j,P_i,P_j,dwdr,er,m)
            #viscous contribution
            F_visc = MorrisViscContrib(mu,rho_i, rho_j,dwdr,rVelViscous,rPos,m)
            # Add the forces contrib
            forces[i,:] = np.sum(F_Pres,0)+np.sum(F_visc,0)+grav
            #continuity contribution
            drhodt[i] = np.sum(velocityDivContrib(rVelCont,rPos,dwdr,er,m),0)
    return forces,drhodt
    
    
    