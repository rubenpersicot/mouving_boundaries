###########################################################################
###########################################################################
#                             SPyH
###########################################################################
###########################################################################
#Authors :  R. Carmigniani & D. Violeau
#Version : SPyH
#Contact : remi.carmigniani@enpc.fr
###########################################################################
# In this file we list all the parameters stored in the array of particles
# and the flags
import numpy as np
## FLAGS :
# Type of particles
FLUID = 0
BOUND = 1
MOBILEBOUND = 2
MOBILESOLID = 3
#
nBound = 4
#
#KERNEL PARAMETERS
aW = 7/(4*np.pi)
d=2
smthfc=2
nparameters=10
POS = [0,1]
VEL = [2,3]
RHO = 4
FORCES = [5,6]
DRHODT = 7
SPID = 8
INFO = 9
#
maxSpaceNeibs = 9
maxPartInSpace = 25
