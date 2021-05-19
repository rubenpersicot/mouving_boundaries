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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('text', usetex=True)
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
import matplotlib.colorbar as cbar
import matplotlib.cm
from src.spyh import *
from src.sphvar import *

def plotPropertiesWithBound(part,partProp,nameBar,bounds,dr,PARTFIG):
    '''
   input : 
        -part : list of particles
        -partProp : array of data to display
        -nameBar : legend for the bar color
        -bounds : 
            [xMin,xMax,yMin,yMax,propMin, propMax] the bound of the domain and color bar
        -PARTFIG : the ID of the plot
    output : 
        -display a png image of the simulation but does not save it
    '''
    if len(bounds)==6:
       xMin = bounds[0]
       xMax = bounds[1]
       yMin = bounds[2]
       yMax = bounds[3]
       propMin = bounds[4]
       propMax = bounds[5]
    else: 
       print('Error the bounds should have 6 inputs!!! check plotParticles function')
       return
    #Create a colormap
    def f_color_map(x):
        Nstep = 100
        jet = matplotlib.cm.get_cmap('jet',Nstep)
        return jet((x-propMin)/(propMax-propMin))
    cmap = plt.cm.jet
    normal = plt.Normalize(propMin,propMax) # my numbers from 0-1
    infoTab = part[:,INFO]
    cnts = part[infoTab==FLUID][:,POS]
    offs = np.ones([4,len(cnts),2])
    offs[0,:,:] = [-dr/2,-dr/2]
    offs[1,:,:] = [dr/2,-dr/2]
    offs[2,:,:] = [dr/2,dr/2]
    offs[3,:,:] = [-dr/2,dr/2]
    vrts_fluid = cnts + offs
    vrts_fluid = np.swapaxes(vrts_fluid, 0, 1)
    cnts = part[infoTab==BOUND][:,POS]
    offs = np.ones([4,len(cnts),2])
    offs[0,:,:] = [-dr/2,-dr/2]
    offs[1,:,:] = [dr/2,-dr/2]
    offs[2,:,:] = [dr/2,dr/2]
    offs[3,:,:] = [-dr/2,dr/2]
    vrts_bound = cnts + offs
    vrts_bound = np.swapaxes(vrts_bound, 0, 1)
    #MOBILE BOUNDARIES PROPERTIES 
    cnts = part[infoTab==MOBILEBOUND][:,POS]
    offs = np.ones([4,len(cnts),2])
    offs[0,:,:] = [-dr/2,-dr/2]
    offs[1,:,:] = [dr/2,-dr/2]
    offs[2,:,:] = [dr/2,dr/2]
    offs[3,:,:] = [-dr/2,dr/2]
    vrts_mobilebound = cnts + offs
    vrts_mobilebound = np.swapaxes(vrts_mobilebound, 0, 1)
    #MOBILE SOLIDS PROPERTIES 
    cnts = part[infoTab==MOBILESOLID][:,POS]
    offs = np.ones([4,len(cnts),2])
    offs[0,:,:] = [-dr/2,-dr/2]
    offs[1,:,:] = [dr/2,-dr/2]
    offs[2,:,:] = [dr/2,dr/2]
    offs[3,:,:] = [-dr/2,dr/2]
    vrts_mobilesolid = cnts + offs
    vrts_mobilesolid = np.swapaxes(vrts_mobilesolid, 0, 1)
    
    rgb = f_color_map(partProp[infoTab==FLUID])
    # create the figure
    fig = plt.figure(PARTFIG)
    ax_list = fig.axes#check if the figure is already open else get back the subplot axes
    if len(ax_list)<1:
    	ax = fig.subplots()
    else:
    	ax = ax_list[0]
    coll = PolyCollection(vrts_fluid,array=None,facecolors=rgb,edgecolor='black',linewidths=0.1)
    ax.add_collection(coll)
    coll = PolyCollection(vrts_bound,array=None,facecolors='gray',edgecolor='black',linewidths=0.1)
    ax.add_collection(coll)
    coll = PolyCollection(vrts_mobilebound,array=None,facecolors='white',edgecolor='black',linewidths=0.1)
    ax.add_collection(coll)
    coll = PolyCollection(vrts_mobilesolid,array=None,facecolors='purple',edgecolor='black',linewidths=0.1)
    ax.add_collection(coll)
    ax.set_aspect('equal') 
    plt.xlabel('$x$(m)',fontsize=18)
    plt.ylabel('$y$(m)',fontsize=18)
    plt.xlim(xMin,xMax)
    plt.ylim(yMin,yMax) 
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cax, _ = cbar.make_axes(ax) 
    cb2 = cbar.ColorbarBase(cax, cmap=cmap,norm=normal)
    cb2.set_label(nameBar,fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    ##fig.savefig(figname,bbox_inches='tight')
    plt.show(block=False)
    plt.draw()

def particleOutline(part,partID,color,dr,PARTFIG):
# 	input : 
#		-part : list of particles
#		-PartID : np.array of the particles to display
#		-color : the color of the edge to outline
#		-PARTFIG : the ID of the plot 
#	output : 
#		-display a png image of the simulation but does not save it
	fig = plt.figure(PARTFIG)
	ax = fig.axes
	ax = ax[0]
	fluidPart = part[partID][:,POS]
	cnts = fluidPart[:]
	offs = np.ones([4,len(fluidPart),2])
	offs[0,:,:] = [-dr/2,-dr/2]
	offs[1,:,:] = [dr/2,-dr/2]
	offs[2,:,:] = [dr/2,dr/2]
	offs[3,:,:] = [-dr/2,dr/2]
	vrts_fluid = cnts + offs
	vrts_fluid = np.swapaxes(vrts_fluid, 0, 1)
	coll = PolyCollection(vrts_fluid,array=None,facecolors='none',edgecolor=color,linewidths=3)
	ax.add_collection(coll)
	plt.show(block=False)
	plt.draw()

def plotSpaces(posSpace,color,lspace,PARTFIG):
# 	input : 
#		-space : list of particles
#		-color : the ID of the plot
#		-PARTFIG : the ID of the plot 
#	output : 
#		-display a png image of the simulation but does not save it
	
	fig = plt.figure(PARTFIG)
	ax = fig.axes
	ax = ax[0]
	cnts = posSpace
	offs = np.ones([4,len(posSpace),2])
	offs[0,:,:] = [-lspace/2,-lspace/2]
	offs[1,:,:] = [lspace/2,-lspace/2]
	offs[2,:,:] = [lspace/2,lspace/2]
	offs[3,:,:] = [-lspace/2,lspace/2]
	vrts_space = cnts + offs
	vrts_space = np.swapaxes(vrts_space, 0, 1)
	coll = PolyCollection(vrts_space,array=None,facecolors='none',edgecolor=color,linewidths=1)
	ax.add_collection(coll)
	for i in range(len(posSpace)):
		ax.text(posSpace[i,0], posSpace[i,1],r''+repr(i), fontsize=14,horizontalalignment='center',verticalalignment='center')
	plt.show(block=False)
	plt.draw()

def spacesOutline(posSpace,color,lspace,PARTFIG):
# 	input : 
#		-space : list of particles
#		-color : the ID of the plot
#		-PARTFIG : the ID of the plot 
#	output : 
#		-display a png image of the simulation but does not save it
	
	fig = plt.figure(PARTFIG)
	ax = fig.axes
	ax = ax[0]
	cnts = posSpace
	offs = np.ones([4,len(posSpace),2])
	offs[0,:,:] = [-lspace/2,-lspace/2]
	offs[1,:,:] = [lspace/2,-lspace/2]
	offs[2,:,:] = [lspace/2,lspace/2]
	offs[3,:,:] = [-lspace/2,lspace/2]
	vrts_space = cnts + offs
	vrts_space = np.swapaxes(vrts_space, 0, 1)
	coll = PolyCollection(vrts_space,array=None,facecolors='none',edgecolor=color,linewidths=1)
	ax.add_collection(coll)
	plt.show(block=False)
	plt.draw()
	
	
    
def quiverPlot(part,sc,PARTFIG):
# 	input : 
#		-part : list of particles
#		-sc : scale of the vector
#		-PARTFIG : the ID of the plot
#	output : 
#		-display a png image of the simulation but does not save it
    # create the figure
    fig = plt.figure(PARTFIG)
    ax_list = fig.axes#check if the figure is already open else get back the subplot axes
    if len(ax_list)<1:
    	ax = fig.subplots()
    else:
    	ax = ax_list[0]
    x=part[:,POS[0]]
    y=part[:,POS[1]]
    u=part[:,VEL[0]]
    v=part[:,VEL[1]]
    ax.quiver(x,y,u,v,scale=sc,scale_units='inches')
    ##fig.savefig(figname,bbox_inches='tight')
    plt.show(block=False)
    plt.draw()
    
    
def plotProperties(part,partProp,nameBar,bounds,dr,PARTFIG):
    '''
   input : 
        -part : list of particles
        -partProp : array of data to display
        -nameBar : legend for the bar color
        -bounds : 
            [xMin,xMax,yMin,yMax,propMin, propMax] the bound of the domain and color bar
        -PARTFIG : the ID of the plot
    output : 
        -display a png image of the simulation but does not save it
    '''
    if len(bounds)==6:
       xMin = bounds[0]
       xMax = bounds[1]
       yMin = bounds[2]
       yMax = bounds[3]
       propMin = bounds[4]
       propMax = bounds[5]
    else: 
       print('Error the bounds should have 6 inputs!!! check plotParticles function')
       return
    #Create a colormap
    def f_color_map(x):
        Nstep = 100
        jet = matplotlib.cm.get_cmap('jet',Nstep)
        return jet((x-propMin)/(propMax-propMin))
    cmap = plt.cm.jet
    normal = plt.Normalize(propMin,propMax) # my numbers from 0-1
    infoTab = part[:,INFO]
    cnts = part[:,POS]
    offs = np.ones([4,len(cnts),2])
    offs[0,:,:] = [-dr/2,-dr/2]
    offs[1,:,:] = [dr/2,-dr/2]
    offs[2,:,:] = [dr/2,dr/2]
    offs[3,:,:] = [-dr/2,dr/2]
    vrts_fluid = cnts + offs
    vrts_fluid = np.swapaxes(vrts_fluid, 0, 1)
    
    rgb = f_color_map(partProp)
    # create the figure
    fig = plt.figure(PARTFIG)
    ax_list = fig.axes#check if the figure is already open else get back the subplot axes
    if len(ax_list)<1:
    	ax = fig.subplots()
    else:
    	ax = ax_list[0]
    coll = PolyCollection(vrts_fluid,array=None,facecolors=rgb,edgecolor='black',linewidths=0.1)
    ax.add_collection(coll)
    ax.set_aspect('equal') 
    plt.xlabel('$x$(m)',fontsize=18)
    plt.ylabel('$y$(m)',fontsize=18)
    plt.xlim(xMin,xMax)
    plt.ylim(yMin,yMax) 
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cax, _ = cbar.make_axes(ax) 
    cb2 = cbar.ColorbarBase(cax, cmap=cmap,norm=normal)
    cb2.set_label(nameBar,fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    ##fig.savefig(figname,bbox_inches='tight')
    plt.show(block=False)
    plt.draw()
