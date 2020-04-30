# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:02:58 2020

@author: George Willingham
"""


import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .rigid_ion import RigidIon
from .coulomb import Coulomb
from .greens_function import GreensFunction




class Model:
    """
    Primary class for lattice dynamics. Constructs the model and allows for
    diagonalizing the full dynamical matrix.
    
    Parameters
    ----------
    lattice : Lattice object
              Instance of Lattice for the crystal of interest
    couplingArray : 2D array
                    2D array of coupling constants to be used in rigid ion
                    calculation
    threshold : float
                Radius of sphere containing all of an atom's neighbors
    coulomb : Bool, optional
              To include or not include Coulomb calculation in dynamical 
              matrix
    charges : array_like, optional
              Array/list containing the charges of atoms in unit cell in terms
              of electron charge. Must respect order of unit cell atoms in
              Lattice object
    GSumDepth : int, optional
                Depth of reciprocal lattice sum in Ewald summation
    RSumDepth : int,optional
                Depth of direct lattice sum in Ewald summation
    eta : float, optional
          The integral-splitting factor for Ewald summation
    """
    
    def __init__(self, 
                 lattice,
                 couplingArray,
                 threshold,
                 coulomb=False,
                 charges=None,
                 GSumDepth=5,
                 RSumDepth=5,
                 eta='default'):
        
        self.lattice = lattice
        self.M = lattice.getMassMatrix()
        self.rigidIon = RigidIon(lattice, 
                              couplingArray, 
                              threshold)
        if coulomb:
            self.withCoulomb=True
            self.coulomb = Coulomb(lattice, 
                                 charges, 
                                 GSumDepth, 
                                 RSumDepth, 
                                 eta)
            self.Z = self.coulomb.Z
            self.C = lambda q: self.coulomb.C(q)
            self.ZCZ = lambda q: self.Z @ self.C(q) @ self.Z
        else:
            self.withCoulomb=False
        
        self.neighbors = self.rigidIon.neighbors
        self.R = lambda q: self.rigidIon.R(q)
        
        if self.withCoulomb:
            self.D = lambda q: self.M @ ( self.R(q) + self.ZCZ(q) ) @ self.M
        else:
            self.D = lambda q: self.M @ self.R(q) @ self.M
        
        self.G = GreensFunction(self.D)
    
     
    
    def getDispersion(self, 
                      qMarkers,
                      pointDensity=35,
                      getNormalModes=False,
                      keepCoulomb=False,
                      showProgress=True,
                      save=True):
        """
        Calculate phonon dispersion along given path through reciprocal space

        Parameters
        ----------
        qMarkers : array_like
                   Array/list of high symmetry points defining path
        pointDensity : int, optional
                       Density of points to sample along path.
                       The default is 35.
        getNormalModes : Bool, optional
                          Whether to return the mass-scaled eigenvectors. 
                          The default is False.
        keepCoulomb : bool, optional
                      If True, the coulomb contribution to the dynamical 
                      matrix is calculated for the given path and stored. 
                      Next time the method is run for the same path, the 
                      stored Coulomb matrices will be used to save CPU time.
                      Default is False
        showProgress : bool, optional
                      Prints the progress of the building Coulomb matrices
                      for storing. Only hass effect if keepCoulomb==True
                      Default is False
        save : bool, optional
               If true, the dispersion, normal modes, q-path, and q-path parts
               are saved as attributes to the Model object

        Returns
        -------
        dispersion : ndarray
                     List containing the calculating phonon frequencies over
                     each q in the path
        normalModes : ndarray, conditional
                      2D array containing the mass-scaled eigenvectors at 
                      each q. The eigenvectors are the COLUMNS of the array.
                      Only returned if getNormalModes==True
        qPath : ndarray
                Array containing all q vectors calculated for
        qPathParts : list
                     Array containing the lines along which the 
                     calculation traveled. (useful for plotting options)
        """
        qPath, qPathParts = self._buildPath(qMarkers, pointDensity)
        
        if keepCoulomb:
            if not hasattr(self, 'storedCoulomb'):
                print('No Coulomb matrices stored. Making them now:')
                self.storedCoulomb = {}
                progress = 0
                for q in qPath:
                    self.storedCoulomb[str(q)] = self.Z @ self.C(q) @ self.Z 
                    if showProgress:
                        progress += 1
                        print(f'\r{progress}/{len(qPath)}',end='')
                if showProgress: print('\nCoulomb matrices stored')
                
            M = self.M
            R = self.R
            DList = [M @ ( R(q) + self.storedCoulomb[str(q)] )  @ M
                                             for q in qPath]

        else:
            DList = [ self.D(q) for q in qPath ]
            
        dispersion = []
        normalModes = []
        
        if getNormalModes == True:
            for D in DList:
                eigenvalues, eigenvectors = la.eigh(D)
                eigenvalues = np.round(eigenvalues, 10)
                frequencies_THz = np.round(np.sqrt(eigenvalues)/(2*np.pi), 10)
                dispersion.append(frequencies_THz)
                qModes = self.M @ eigenvectors # account for mass normalization
                normalModes.append(qModes)
        else:
            for D in DList:
                eigenvalues = la.eigvalsh(D) 
                eigenvalues = np.round(eigenvalues, 10)
                frequencies_THz = np.round(np.sqrt(eigenvalues)/(2*np.pi) ,10)
                dispersion.append(frequencies_THz)

                
        dispersion = np.array(dispersion)
        normalModes = np.array(normalModes)
        if save==True:
            self.dispersion = dispersion
            self.qPath = qPath
            self.qPathParts = qPathParts
            self.normalModes = normalModes
        
        return dispersion, normalModes, qPath, qPathParts
        
    
    
    def _buildLine(self, 
                   start, 
                   end, 
                   pointDensity):
        """
        Get line from start to end sampled at pointDensity

        Parameters
        ----------
        start : array_like
                Start of line
        end : array_like
              End of line
        pointDensity : int
                       Density of samples along line

        Returns
        -------
        line : ndarray
               List of wavevectors from start to end

        """
        
        start = np.array(start)
        end = np.array(end)
        numPoints = int((pointDensity * la.norm( start - end )))
        samples = np.linspace(0, 1, numPoints)
        line = np.array( [ (1-x)*start + x*end  for x in samples] )

        return line
    

    def _buildPath(self,
                   qMarkers,
                   pointDensity):
        """
        Get path in reciprocal space which travels between qMarkers

        Parameters
        ----------
        qMarkers : array_like
                   Array/list of high symmetry points defining the path
        pointDensity : int
                       Density of samples along path

        Returns
        -------
        qPath : ndarray
                Array/list of wavevectors in path
        qPathParts : list
                     Array/list of lines in path
        """
        
        qPathParts = []
        firstLine = self._buildLine(qMarkers[0], qMarkers[1], pointDensity)
        qPath = firstLine
        qPathParts.append(firstLine)

        for i in range(1, len(qMarkers)-1):
            start = qMarkers[i]
            end = qMarkers[i+1]
            line = self._buildLine(start, end, pointDensity)
            qPath = np.append(qPath, line[1::], axis=0)
            qPathParts.append(line)
        
        return qPath, qPathParts
    
    
    def plotDispersion(self,
                       labels=[],
                       figsize=(15,6),
                       title='',
                       style='r-',
                       markersize=5,
                       ylim=[0, None],
                       withSurfaceModes=False,
                       surfaceStyle='k.',
                       surfaceMarkersize=5,
                       withDOS=True,
                       binDensity=60,
                       smoothenDOS=True,
                       sigma=0.5,
                       normalize=True):
        
        dispersion = self.dispersion
        qPath = self.qPath
        qPathParts = self.qPathParts
        
        if withDOS:
            DOS, bins = self.getDOS(dispersion,
                              binDensity=binDensity,
                              smoothen=smoothenDOS,
                              sigma=sigma,
                              normalize=normalize)
            gridspec_kw = {'hspace':0,
                           'wspace':0.03,
                           'width_ratios':[4,1]}
        else:
            gridspec_kw = {'hspace':0,
                           'wspace':0,
                           'width_ratios':[4,0]} # tucks away the DOS axes
            
        f, (ax, axDOS) = plt.subplots(1,2, 
                             figsize=figsize,
                             sharey='row',
                             gridspec_kw=gridspec_kw)
        params = {
                  'axes.labelsize': 18,
                  'axes.titlesize': 22,
                  'xtick.labelsize' :22,
                  'ytick.labelsize': 18,
                  'grid.color': 'k',
                  'grid.linestyle': ':',
                  'grid.linewidth': 0.5,
                  'mathtext.fontset' : 'stix',
                  'mathtext.rm'      : 'serif',
                  'font.family'      : 'serif',
                  'font.serif'       : "Times New Roman"        
                 }
        mpl.rcParams.update(params)
    
        ax.plot(dispersion, style, markersize=markersize)
        ax.set_title(title)
        ax.set_ylabel('$\\nu$ (THz)', rotation=90, labelpad=20)

        ax.set_xlim(0, len(qPath)-1)
        ax.set_ylim(ylim[0], ylim[1])
    
        # set xticks
        lineLengths = [0]+[len(qLine)-1 for qLine in qPathParts]
        tick_locs = np.cumsum(lineLengths)
        ax.set_xticks( tick_locs )
        ax.set_xticklabels( labels )
        for tick in tick_locs:
            ax.axvline(tick, color='k', alpha=0.3)
        ax.grid(False)
        
        
        if withSurfaceModes:
            ax.plot(self.surfaceDispersion, 
                    surfaceStyle, 
                    markersize=surfaceMarkersize)


        axDOS.set_xticklabels([])
        if withDOS:
            axDOS.fill_between(DOS, bins[:-1], color='k', alpha=0.5)
            axDOS.set_xlim(0, max(DOS)*1.05)
            axDOS.grid(True)
            axDOS.set_title('DOS')
        
        plt.show()

    
    
    def getDOS(self,
               dispersion=[],
               binDensity=60,
               smoothen=False,
               sigma=0.05,
               normalize=True
               ):
        """
        Finds the relative DOS integrated over the path through the 
        Brillouin zone.

        Parameters
        ----------
        dispersion : list, optional
                     Dispersion. The default is [].
        binDensity : int, optional
                     Density of bins along energy axis. 
                     The default is 60.
        smoothen : bool, optional
                   Bool for whether the DOS should be smoothened or not. 
                   The default is False.
        sigma : float, optional
                Width of gaussian blur used if smoothen==True. 
                The default is 0.05.
        normalize : bool, optional
                    Bool for whether to normalize the determined relative DOS.
                    The default is True.

        Returns
        -------
        histogram : list
                    List of relative DOS values.
        bins : list
               Bins used.

        """
        
        if dispersion==[]: 
            dispersion = self.dispersion
            
        dispersion = dispersion.flatten()
        topOfRange = max(dispersion)*1.05
        numBins = int(topOfRange*binDensity)
        bins = np.linspace(0, topOfRange, numBins)
        
        histogram, bins = np.histogram(dispersion, bins)
        
        if smoothen:
            histogram = gaussian_filter1d(histogram, sigma)
        
        if normalize:
            histogram = histogram/max(histogram)
        
        return histogram, bins
        
        
        
    def getProjectedDispersion(self,
                          surfaceMarkers,
                          zMarkers,
                          pointDensity=35,
                          zPointDensity=15):
        """
        Calculate the bulk disperion projected onto a particular direction.

        Parameters
        ----------
        surfaceMarkers : list
                        A list of high symmetry points in the desired surface
                        Brillouin zone. 
        zMarkers : list of length=2
                   List of q-vectors normal to the desired surface between 
                   which the projections will be calculated.
        pointDensity : int, optional
                    The density of sampled points along path through surface
                    Brillouin zone.
                    The default is 35.
        zPointDensity : The density of sampled points along projection axis, optional
                        The default is 15.

        Returns
        -------
        projectionLayers : list
                           List of 2D Dispersions calculated along planes 
                           parallel to the surface of interest
        """
        
        qzPath, qzPathParts = self._buildPath(zMarkers, zPointDensity)
        
        projectionLayers = []        
        for qz in qzPath:
            qMarkers = [np.array(q)+qz for q in surfaceMarkers]
            results = self.getDispersion(qMarkers,
                                         pointDensity,
                                         getNormalModes=False,
                                         keepCoulomb=False,
                                         showProgress=False,
                                         save=False)
            projectionLayers.append(results[0])
            
        self.projectedDispersion = projectionLayers
        self.surfPath, self.surfPathParts = self._buildPath(surfaceMarkers,
                                                            pointDensity)
        
        return projectionLayers
    
    
    
    def plotProjectedDispersion(self,
                                labels=[],
                                figsize=(16,8),
                                title='',
                                style='r-',
                                markersize=5,
                                ylim=[0, None]):
        """
        Plot the bulk projected dispersion.

        Parameters
        ----------
        See plotDispersion method. They are the same
        """
        projectedDispersion = self.projectedDispersion
        qPath = self.surfPath
        qPathParts = self.surfPathParts
        
        f, ax = plt.subplots(figsize=figsize)
        params = {
                  'axes.labelsize': 18,
                  'axes.titlesize': 22,
                  'xtick.labelsize' :22,
                  'ytick.labelsize': 18,
                  'grid.color': 'k',
                  'grid.linestyle': ':',
                  'grid.linewidth': 0.5,
                  'mathtext.fontset' : 'stix',
                  'mathtext.rm'      : 'serif',
                  'font.family'      : 'serif',
                  'font.serif'       : "Times New Roman"        
                 }
        mpl.rcParams.update(params)
        
        for layer in projectedDispersion:
            _plotLayer = ax.plot(layer, style, markersize=markersize)
        ax.set_title(title)
        ax.set_ylabel('$\\nu$ (THz)', rotation=90, labelpad=20)

        ax.set_xlim(0, len(qPath)-1)
        ax.set_ylim(ylim[0], ylim[1])
        
        lineLengths = [0]+[len(qLine)-1 for qLine in qPathParts]
        tick_locs = np.cumsum(lineLengths)
        ax.set_xticks( tick_locs )
        ax.set_xticklabels( labels )
        for tick in tick_locs:
            ax.axvline(tick, color='k', alpha=0.3)
        ax.grid(False)
        plt.show()
            
        
        
    def _isSurfaceMode(self, 
                       mode, 
                       weightThreshold, 
                       numLayers,
                       method='atomic'):
        """
        Returns bool whether given mode qualifies as a surface mode.

        Parameters
        ----------
        mode : arraylike
            normal mode array
        weightThreshold : float
                        percentage of oscillator weight which must be
                        within numLayers of surface to qualify as surface mode
        numLayers : int
                    Number of unit cells to include in surface
        method : str, optional
                Method for calculating oscillator strength.
                'atomic' - takes the norms of each atom's true oscillation
                amplitude.
                'abs' - takes absolute values of normal mode and uses all
                components in calculation.
                The default is 'atomic'.

        Returns
        -------
        bool
            Is the mode a surface mode?

        """
        
        if method=='abs':
            checkDepth = 3*self.lattice.bulk.atomsPerUnitCell * numLayers
            absMode = abs(mode)
            total = sum(absMode)
            bottomSurfaceWeight = sum(absMode[ : checkDepth]) / total
            topSurfaceWeight = sum(absMode[-checkDepth : ]) / total
            
        elif method=='atomic':
            checkDepth = self.lattice.bulk.atomsPerUnitCell * numLayers
            atomicMotion = [la.norm(mode[i:i+3]) 
                            for i in range(0, len(mode), 3)]
            total = sum(atomicMotion)
            bottomSurfaceWeight = sum(atomicMotion[ : checkDepth]) / total
            topSurfaceWeight = sum(atomicMotion[ -checkDepth : ]) / total
        
        elif method=='prob':
            M_inv = la.inv(self.M)
            mode = M_inv @ mode
            checkDepth = self.lattice.bulk.atomsPerUnitCell * numLayers
            probabilities = [la.norm(mode[i:i+3])**2 
                            for i in range(0, len(mode), 3)]
            total = sum(probabilities)
            bottomSurfaceWeight = sum(probabilities[ : checkDepth]) / total
            topSurfaceWeight = sum(probabilities[ -checkDepth : ]) / total
            
        
        surfaceWeight = bottomSurfaceWeight + topSurfaceWeight
        
        return surfaceWeight >= weightThreshold
        
        
        
        
    def getSurfaceModes(self,
                        weightThreshold=0.3,
                        numLayers=3,
                        method='atomic',
                        save=True):
        """
        Gets the surface modes from the already calculated normal modes in the
        model.

        Parameters
        ----------
        weightThreshold : float
                        Percentage of oscillator weight which must be
                        within numLayers of surface to qualify as surface mode
        numLayers : int
                    Number of unit cells to include in surface
        method : str, optional
                Method for calculating oscillator strength.
                'atomic' - takes the norms of each atom's true oscillation
                amplitude.
                'abs' - takes absolute values of normal mode and uses all
                components in calculation.
                The default is 'atomic'.
        save : bool, optional
                Whether or not to store the determined surface modes/dispersion. 
                The default is True.

        Returns
        -------
        surfaceModes : array
                        Array containing the surface modes.
        surfaceDispersion : array
                            Array containing the surface dispersion

        """
        
        if not hasattr(self.lattice, 'slabCell'):
            raise AttributeError('Only slab model has surface modes')
        if not hasattr(self, 'normalModes'):
            raise AttributeError('No eigenvectors stored')
        
        isSurfaceMode = lambda mode: self._isSurfaceMode(mode, 
                                                         weightThreshold, 
                                                         numLayers,
                                                         method=method)
        surfaceModes = []    
        _surfaceModesInx = []
        surfaceDispersion = []
        
        for qi in range(len(self.qPath)):
            surfaceModes.append([])
            _surfaceModesInx.append([])
            surfaceDispersion.append([])
            qModes = self.normalModes[qi]
            
            for j in range(qModes.shape[0]):
                mode = qModes[:,j] # recall the modes are the columns of this 2D array
                
                if isSurfaceMode(mode):
                    surfaceModes[qi].append( mode )
                    _surfaceModesInx[qi].append( j )
                    surfaceDispersion[qi].append( self.dispersion[qi, j] )
                else:
                    surfaceDispersion[qi].append(None)
                    
        surfaceModes = np.array(surfaceModes)
        surfaceDispersion = np.array(surfaceDispersion)
        
        if save:
            self.surfaceModes = surfaceModes
            self._surfaceModesInx = _surfaceModesInx
            self.surfaceDispersion = surfaceDispersion
            
        return surfaceModes, surfaceDispersion
                
            
    
    
    
            
            
        
        
        