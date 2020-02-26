# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:02:58 2020

@author: George Willingham
"""


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib

from .rigid_ion import RigidIon
from .coulomb import Coulomb




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
          Default is inverse cube root of lattice cell volume
    """
    
    def __init__(self, 
                 lattice,
                 couplingArray,
                 threshold,
                 coulomb=True,
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
        else:
            self.withCoulomb=False
        
        self.neighbors = self.rigidIon.neighbors
        self.R = lambda q: self.rigidIon.R(q)
        self.D = lambda q: self.getDynamicalMatrix(q)
    
    
    def getDynamicalMatrix(self, 
          q):
        """
        Get full dynamical matrix at wavevector q

        Parameters
        ----------
        q : array_like
            Wavevector to calculate dynamical matrix

        Returns
        -------
        _D : numpy matrix
             Full dynamical matrix at wavevector q
        """
        q = np.array(q)
        M = self.M
        R = self.R
        modelParts = R(q)
        if self.withCoulomb:
            modelParts += self.Z @ self.C(q) @ self.Z
            
        _D = M @ modelParts @ M
        
        return _D
        
        
    
    def getDispersion(self, 
                      qMarkers,
                      pointDensity=35,
                      getEigenVectors=False,
                      keepCoulomb=False,
                      showProgress=False):
        """
        Calculate phonon dispersion along given path through reciprocal space

        Parameters
        ----------
        qMarkers : array_like
                   Array/list of high symmetry points defining path
        pointDensity : int, optional
                       Density of points to sample along path.
                       The default is 35.
        getEigenVectors : Bool, optional
                          Whether to return the eigenvectors. 
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

        Returns
        -------
        dispersion : ndarray
                     List containing the calculating phonon frequencies over
                     each q in the path
        normalModes : ndarray, conditional
                      Array containing the eigenvectors at each q.
                      Only returned if getEigenVectors==True
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
            self.withCoulomb = False # temporarily set withCoulomb to False
            DList = [self.D(q) + M @ self.storedCoulomb[str(q)] @ M 
                                             for q in qPath]
            self.withCoulomb = True # reset withCoulomb to True
        else:
            DList = [ self.D(q) for q in qPath ]
            
        dispersion = []
        normalModes = []
        
        for D in DList:
            if getEigenVectors==True:
                [eigenValues, eigenVectors] = la.eigh(D)
                eigenValues = np.round(eigenValues, 10)
                frequencies_THz = np.round(np.sqrt(eigenValues)/(2*np.pi), 10)
                dispersion.append(frequencies_THz)
                normalModes.append(eigenVectors)
            else:
                eigenValues = la.eigvalsh(D) 
                eigenValues = np.round(eigenValues, 10)
                frequencies_THz = np.round(np.sqrt(eigenValues)/(2*np.pi) ,10)
                dispersion.append(frequencies_THz)

                
        dispersion = np.array(dispersion)
        normalModes = np.array(normalModes)
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
                       figsize=(16,8),
                       title='',
                       style='r-',
                       markersize=5,
                       ylim=[0, None]):
        dispersion = self.dispersion
        qPath = self.qPath
        qPathParts = self.qPathParts
        
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
        matplotlib.rcParams.update(params)
    
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
        plt.show()