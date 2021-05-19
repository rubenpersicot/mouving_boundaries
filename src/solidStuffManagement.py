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

def computeCenterOfMass(part, n):
    '''
    -part : ....
    - n : number of solid particle
    
    '''
    infoTab = part[:,INFO]
    OG = np.array([1/n*np.sum(part[infoTab == MOBILESOLID] [:,POS[0]]),1/n*np.sum(part[infoTab == MOBILESOLID] [:,POS[1]])])
    return OG

    