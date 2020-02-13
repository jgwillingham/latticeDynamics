# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:25:08 2020

This module contains the classes relevant to calculating
phonon dispersion for materials. The Lattice class inherits from 
the Crystal class which provides symmetry operations and more.

work in progress


@author: George Willingham
"""

import numpy as np
import scipy.linalg as la
from scipy.special import erfc 
from crystals import Atom, Crystal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib




class Lattice(Crystal):
    
    """
    Lattice inherits from the Crystal class. Crystal objects contain 
    information about lattice symmetry, reciprocal lattices, etc. 
    Lattice objects additionally find near neighbors and finds mass matrix
    for the dynamical matrix.
    
    Parameters
    ----------
    latticeVectors : array_like
                     Contains the primitive lattice vectors of the lattice 
                     as arrays/lists.
                     
    unitCell : array_like
               Contains information about all atoms in the unit cell. 
               Each entry should be in the form 
               (('SYMBOL'), FRACTIONAL_POSITION) where 
               SYMBOL is the atom's element symbol as a string and 
               FRACTIONAL_POSITION is the atom's fractional position 
               in the unit cell.
               e.g. [('Cd', (0, 0, 0)) , ('Te' , (1/4, 1/4, 1/4)) ]
    
    Notes
    -----
    Be aware of the form of the unitCell parameter! 
    Correct element symbol from the periodic table is needed to 
    obtain the masses. 
    """
    
    def __init__(self,
                 unitCell,
                 latticeVectors):
        self._unitCell = [Atom(atom[0], coords=atom[1]) 
                              for atom in unitCell]
        Crystal.__init__(self, self._unitCell, latticeVectors)
        self.atomsPerUnitCell = len(unitCell)
        self.atomicWeights = [atom.mass for atom in self._unitCell]
        self.atomLabels = [f'{inx}_{atom.element}' 
                      for atom, inx in 
                      zip(self._unitCell, range(self.atomsPerUnitCell))]
        
        
    def getNeighbors(self, 
                     threshold, 
                     cellSearchWidth=1):
        """
        Get lists of neighbors for each atom in the unit cell

        Parameters
        ----------
        threshold : float
                    Radius of the sphere containing all neighbors
        cellSearchWidth : int, optional
                          Number of lattice cells away to search for neighbors
                          in all 3 dimensions. 
                          The default is 1.

        Returns
        -------
        neighbors : dict
                    Dictionary containing every atom in the unit cell's 
                    neighbors as the key-value pairs: SYMBOL-NEIGHBOR_LIST.
                    Neighbor lists are accessed with element name given in
                    the unitCell parameter when Lattice object is instantiated.
                    e.g. 
        """
        
        searchWidth = range(-cellSearchWidth, cellSearchWidth+1)
        neighbors = {atomLabel :[] for atomLabel in self.atomLabels}
        
        for atom_i, label_i in zip(self._unitCell , self.atomLabels):
            Ri = atom_i.coords_cartesian
            
            for atom_j, label_j in zip(self._unitCell , self.atomLabels):
                xj = atom_j.coords_cartesian
                
                for s1 in searchWidth:
                    for s2 in searchWidth:
                        for s3 in searchWidth:
                            
                            latVec = s1*self.a1 + s2*self.a2 + s3*self.a3
                            Rj = xj + latVec
                            bond_ij = Rj - Ri
                            distance_ij = la.norm(bond_ij)
                            
                            if distance_ij != 0 and distance_ij <= threshold:
                                neighbors[label_i].append( 
                                    ( (label_i, label_j) , bond_ij ) 
                                    )
        self.neighbors = neighbors
  
        return neighbors
        
    
    def getMassMatrix(self):
        """
        Gets the diagonal mass matrix used in the dynamical matrix.

        Returns
        -------
        M : numpy matrix
            Diagonal matrix containing elements of the form 1/sqrt(m) 
        """
        NAvagadro = 6.02*10**23 # Avagadro's number
        # get masses in 10**-24 kg = 10**-21 g
        massList = [atomicWeight/NAvagadro *(10**21) 
                        for atomicWeight in self.atomicWeights]
        
        massDiagonal = [ [1/np.sqrt(m)]*3 for m in massList] # note [i]*3 = [i,i,i]
        massDiagonal = np.array(massDiagonal).flatten()
        M = np.diag(massDiagonal)
        M = np.matrix(M)
        
        return M

    def visualizeNeighbors(self, 
                           atomLabel, 
                           atomSize=600):
        """
        Plots the neighbors of atom in 3D space.

        Parameters
        ----------
        atom : string
               Element symbol as string just as given in instantiation of 
               Lattice object
        atomSize : float, optional
                   Size of neighbors appearing in plot. 
                   The default is 600.

        """
        
        cmap = plt.cm.RdGy
        color_id = np.linspace(0, 1, self.atomsPerUnitCell)
        
        atom_neighbors = self.neighbors[atomLabel]
        
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        
        ax.scatter((0), (0), (0) , s=atomSize, color='y')
        
        for atom, i in zip(self.atomLabels, color_id):
            neighbor_coords = [neighbor[1] for neighbor in atom_neighbors if neighbor[0][1]==atom]
            if len(neighbor_coords) != 0:
                x, y, z = zip(*neighbor_coords)
                ax.scatter( x, y, z, s=atomSize, color=cmap(i), label=atom)
        
        ax.set_axis_off()
        ax.legend()
        plt.show()

#%%


class RigidIon:
    """
    Class containing all functions needed for the rigid ion contributions
    to the dynamical matrix
    
    Parameters
    ----------
    lattice : Lattice object
              An instance of the Lattice class for the desired crystal
    couplingArray : 2D array_like
                    2D array containing the force constants for calculating 
                    the force constant matrices.
                    Element [i,j] should contain the coupling for interactions 
                    between atom_i and atom_j as listed in the Lattice object.
    """
    def __init__(self,  
                 lattice,
                 couplingArray,
                 threshold):
        
        self.lattice = lattice
        self.threshold = threshold
        self.couplings = couplingArray
        self.neighbors = lattice.getNeighbors(threshold)
        self.atomLabels = self.lattice.atomLabels
        self.atomsPerUnitCell = self.lattice.atomsPerUnitCell
        
    
    def _forceConstantMatrix(self, 
                             bond_ij, 
                             A, 
                             B):
        """
        Get the 3x3 force constant matrix between two atoms in the lattice.

        Parameters
        ----------
        bond_ij : array_like
                  Array/list of the vector pointing from atom_i to atom_j
        A : float
            Radial force constant
        B : float
            tangential force constant

        Returns
        -------
        Phi : numpy matrix
              3x3 force constant matrix relating atoms i and j
        """
        
        Phi = np.zeros([3, 3])
        e=15.1891
        A *= (e**2 / (2*self.lattice.volume))
        B *= (e**2 / (2*self.lattice.volume))
        
        for x_i in range(3):
            for x_j in range(3):
                Phi[x_i, x_j] = (A - B)*bond_ij[x_i]*bond_ij[x_j] / (la.norm(bond_ij)**2)
                if x_i == x_j:
                    Phi[x_i, x_j] += B
                
        return Phi
        
    
    def _Rblock(self, 
                i, 
                j, 
                q):
        """
        Get the full block contribution to the dynamical matrix: The Fourier
        transform of the force constant matrix relating atoms of type i and j.

        Parameters
        ----------
        i : int
            Index for atom in the unit cell (row of matrix)
        j : int
            Index for atom in the unit cell (column of matrix)
        q : array_like
            Wavevector where the Fourier transform is calculated

        Returns
        -------
        R_ij : ndarray
               3x3 array containing the block contribution to dynamical matrix
               relating unit cell atoms i and j (row=i, col=j)
        """
        R_ij = np.zeros([3,3] , dtype='complex128')
        atom_i = self.atomLabels[i]
        atom_j = self.atomLabels[j]
        i_neighbors = self.neighbors[atom_i]
        (A, B) = self.couplings[i][j]
        
        for neighbor in i_neighbors:
            if neighbor[0][1] == atom_j:
                bond_ij = neighbor[1]
                Phi_ij = self._forceConstantMatrix(bond_ij, A, B)
                R_ij += Phi_ij*np.exp( 1j *q @ bond_ij )
                
        return R_ij
    
    
    def _Rself(self, 
               i):
        """
        Get the self term for atom i

        Parameters
        ----------
        i : int
            Index of atom in the unit cell

        Returns
        -------
        Ri_self : ndarray
                  3x3 array containing the self term for atom i
        """
        Ri_self =  np.zeros( [3,3] , dtype='complex128')
        Gamma = np.array((0, 0, 0))
        
        for j in range(self.atomsPerUnitCell):
            Ri_self -= self._Rblock(i, j, Gamma)
        
        return Ri_self
        
        
    def R(self, 
          q):
        """
        Get the full rigid ion matrix at wavevector q

        Parameters
        ----------
        q : array_like
            Wavevector where matrix should be calculated

        Returns
        -------
        _R : numpy matrix
             Full block matrix contribution to dynamical matrix from 
             rigid ions.
        """
        n = self.atomsPerUnitCell
        blocks = []
        
        for i in range(n):
            blocks.append([])
            for j in range(n):
                blocks[i].append( self._Rblock(i, j, q) )
                if i == j:
                    blocks[i][j] += self._Rself(i)
                    
                
        _R = np.matrix(np.block( blocks ))

        return _R
    
    
# %%


class Coulomb:
    """
    Class containing all methods needed for calculating the Coulomb
    contribution to the dynamical matrix via the Ewald method.
    
    WORK IN PROGRESS
    
    Parameters
    ----------
    lattice : Lattice object
              Instance of the Lattice class for the lattice of interest
    charges : array_like
              array containing the charges of atoms in the unit cell in 
              terms of electron charge. 
              (entries should respect the order of atoms given to Lattice obj)
    GSumDepth : int
                Depth of reciprocal lattice sum in Ewald summation
    RSumDepth : int
                Depth of direct lattice sum in Ewald summation
    eta : float, optional
          The integral-splitting factor for Ewald summation
          Default is inverse cube root of lattice cell volume
    """
    
    def __init__(self, 
                 lattice, 
                 charges,
                 GSumDepth,
                 RSumDepth,
                 eta='default'):
        
        self.lattice = lattice
        self.charges = np.array(charges)
        self.GSumDepth = GSumDepth
        self.RSumDepth = RSumDepth
        if eta == 'default':
            self.eta = (self.lattice.volume)**(-1/3)
        else:
            self.eta = eta
        self.GList = self._buildList(lattice.reciprocal_vectors,
                                     GSumDepth)
        self.RList = self._buildList(lattice.lattice_vectors,
                                     RSumDepth)
        self.Z = self.getChargeMatrix()


    def C(self, 
          q):
        """
        Get full block Coulomb matrix at wavevector q

        Parameters
        ----------
        q : array_like
            Wavevector where matrix should be calculated

        Returns
        -------
        _C : numpy matrix
             Full block Coulomb matrix at wavevector q
        """
        
        q = np.array(q)
        n = self.lattice.atomsPerUnitCell
        blocks = []
        
        for i in range(n):
            blocks.append([])
            xi = self.lattice._unitCell[i].coords_cartesian
            
            for j in range(n):
                xj = self.lattice._unitCell[j].coords_cartesian
                Delta = xj - xi
                
                Cfar_ij = self._qSpaceSum(Delta, q)
                Cnear_ij = self._realSpaceSum(Delta, q)
                C_ij = Cfar_ij + Cnear_ij
                
                C_ij = C_ij * np.exp(-1j * q @ Delta) #<- why minus sign? Seems to only work when it's there
                blocks[i].append( C_ij )
                if i == j:
                    blocks[i][j] += self._Cself(i)
                   
        _C = np.matrix(np.block( blocks ))

        return _C
    
    
    def _Cself(self, 
               i):
        """
        Get Coulomb self term for atom i

        Parameters
        ----------
        i : int
            Index for atom in the unit cell.

        Returns
        -------
        Ci_self : ndarray
                  2D array containing self term for atom i
        """
        
        Gamma = np.array((0, 0, 0))
        Ci_self = np.zeros([3,3], dtype='complex128')
        xi = self.lattice._unitCell[i].coords_cartesian
        
        for j in range(self.lattice.atomsPerUnitCell):
            Zfactor = self.charges[j] / self.charges[i]
            xj = self.lattice._unitCell[j].coords_cartesian
            Delta = xj - xi
            
            Cfar_ij = self._qSpaceSum(Delta, Gamma)
            Cnear_ij = self._realSpaceSum(Delta, Gamma)
            C_ij = Cfar_ij + Cnear_ij
            
            Ci_self -= Zfactor*C_ij
        
        return Ci_self
            
                 
    def _qSpaceSum(self,
                   Delta,
                   q):
        """
        Reciprocal lattice sum in Ewald summation

        Parameters
        ----------
        Delta : array_like
                Vector pointing between atom locations within unit cell.
        q : array_like
            wavevector

        Returns
        -------
        Cfar_ij : ndarray
                  2D array containing the reciprocal lattice sum
        """

        Delta = np.array(Delta)

        Cfar_ij = np.zeros([3,3], dtype='complex128')
        QGList = [np.array(q+G) for G in self.GList]

        for G in QGList:
            norm = la.norm(G)
            term = np.outer(G, G) / norm**2
            term = term * np.exp(1j * G @ Delta)
            term = term * np.exp(-norm**2 / (4*self.eta**2))
            Cfar_ij += term
        
        Cfar_ij = Cfar_ij * (4*np.pi / self.lattice.volume)
        #Cfar_ij = Cfar_ij * np.exp(- 1j * q @ Delta) <- moved to C method
        
        return Cfar_ij
    
    
    
    def _realSpaceSum(self,
                      Delta,
                      q):
        """
        Direct lattice sum in Ewald summation

        Parameters
        ----------
        Delta : array_like
                Vector pointing between atom locations within unit cell.
        q : array_like
            wavevector

        Returns
        -------
        Cfar_ij : 2D array
                  2D array containing the direct lattice sum
        """
        Cnear_ij = np.zeros([3,3] , dtype='complex128')
        DeltaRList = [R-Delta for R in self.RList]
        
        for dR in DeltaRList:
            norm = la.norm(dR)
            y = self.eta*norm
            t1 = np.outer(dR, dR) / norm**5
            t1 = t1 * (3*erfc(y)  +  1/np.sqrt(np.pi) * (6*y + 4*y**3)*np.exp(-y**2))
            t2 = np.eye(3) / norm**3
            t2 = t2 * ( erfc(y) + 2*y * np.exp(-y**2) / np.sqrt(np.pi) )
            term = t1 - t2
            term = term * np.exp(1j * q @ (dR + Delta))
            Cnear_ij += term
        
        #Cnear_ij = -1* Cnear_ij * np.exp(-1j * q @ Delta)
        
        return -1*Cnear_ij
        
        
    def _buildList(self, 
                    vectors, 
                    sumDepth):
        """
        Build list of vector to be summed over in Ewald summation

        Parameters
        ----------
        vectors : array_like
                  Array/list of primitive 
                  lattice vectors for direct/reciprocal lattice
        sumDepth : int
                   Depth of lattice sum

        Returns
        -------
        List : list
               List of vectors to be summed over

        """
        
        (v1, v2, v3) = vectors
        
        # make list of reciprocal/direct lattice vectors to sum over
        Vec = lambda n1,n2,n3 : n1*v1 + n2*v2 + n3*v3
        sumRange = range(-sumDepth, sumDepth+1)
        List = []
        
        for n1 in sumRange:
            for n2 in sumRange:
                for n3 in sumRange:
                    if n1==n2==n3==0:
                        pass
                    else:
                        Vector = Vec(n1, n2, n3)
                        List.append(Vector)
        return List
        

    def getChargeMatrix(self):
        """
        Get charge matrix Z

        Returns
        -------
        Z : numpy matrix
            Diagonal matrix containing the charges.

        """
        
        e = 15.1891
        chargeDiagonal = [ [e*Z]*3 for Z in self.charges] # note [i]*3 = [i,i,i]
        chargeDiagonal = np.array(chargeDiagonal).flatten()
        Z = np.diag(chargeDiagonal)
        Z = np.matrix(Z)
        
        return Z
    

#%%


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
            self.withCoulomb = False
            DList = [self.D(q) + M @ self.storedCoulomb[str(q)] @ M 
                                             for q in qPath]
            self.withCoulomb=True
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
        if type(start) != np.ndarray and type(start) != np.ndarray: 
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
                  'font.serif'       : "Times New Roman"# or "Times"          
                 }
        matplotlib.rcParams.update(params)
    
        ax.plot(dispersion, style, markersize=2)
        ax.set_title(title)
        ax.set_ylabel('$\\nu$ (THz)', rotation=90, labelpad=20)
        #ax.set_ylim(0, 5)
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
# %%


# =============================================================================
# 
# TESTING GROUND
# 
# =============================================================================

# =============================================================================
# 
# a = 6.60 # lattice constant in Angstroms
#         # Xu et al: 6.60
#         # Experimental 6.48
# Cd_position = (0, 0, 0)
# Te_position = (1/4, 1/4, 1/4)
# charge = 0.441 # values used in Rowe et al. 0.441, 0.27
# 
# a1 = (1/2, 0, 1/2)
# a2 = (1/2, 1/2, 0)
# a3 = (0, 1/2, 1/2) 
# 
# latticeVectors = [a*np.array(ai) for ai in [a1, a2, a3]]
# unitCell = ( 
#              ("Cd" ,  np.array(Cd_position)   ) , 
#              ( "Te" , np.array(Te_position)   ) 
#            )
# charges = [charge, -charge]
# 
# G = 2*np.pi/a
# Gamma = np.array((0, 0, 0))
# X = (0, G, 0)
# L = (G/2, G/2, G/2)
# W = (G/2, G, 0)
# U = (G/4, G, G/4)
# K = (3*G/4, 3*G/4, 0)
# 
# # NN interactions (Cd-Te)
# Ann = -27 ## -25 <- tuning this seems to push the high optical modes around
# Bnn = -0.2 # -0.2 <- moves acoustic modes
# # NNN interactions (Cd-Cd or Te-Te)
# Cd_Annn = -2.9 # -2.9
# Cd_Bnnn = -1.2 # -0.2
# Te_Annn = -4.2 # -4.2
# Te_Bnnn = -1.1 # -0.1
# 
# 
# couplings = [ [(Cd_Annn,Cd_Bnnn), (Ann, Bnn)], 
#               [(Ann, Bnn), (Te_Annn, Te_Bnnn)] 
#             ]
# 
# =============================================================================
#%%


# =============================================================================
# 
# lattice = Lattice(unitCell, latticeVectors)        
# model = Model(lattice, couplings, threshold=5, coulomb=False, charges=charges)
# 
# qMarkers = [Gamma, X, W, K, Gamma, L]
# qLabels = ['$\Gamma$', 'X', 'W', 'K', '$\Gamma$' , 'L']
# dispersion, normalModes, qPath, qPathParts = model.getDispersion(qMarkers, 
#                                                                  20,
#                                                                  True)
# 
# =============================================================================


# %%

# =============================================================================
# =============================================================================
# # 
# # Convergence tests for qSpaceSum and realSpaceSum
# # 
# =============================================================================
# 
# Delta = (0,0,0)
# q = (0, 1, 0)
# eta = (lattice.unitCellVol)**(-1/3)
# 
# 
# def qSpaceSum(
#                Delta,
#                q, 
#                eta,
#                sumdepth):
#     
#     GList = model.C_obj._buildList(lattice.reciprocalLatticeVectors, sumdepth)
#     Delta = np.array(Delta)
#     q = np.array(q)
# 
#     Cfar_ij = np.zeros([3,3], dtype='complex128')
#     QGList = [np.array(q+G) for G in GList]
# 
#     for G in QGList:
#         norm = la.norm(G)
#         term = np.outer(G, G) / norm**2
#         term = term * np.exp(1j * G @ Delta)
#         term = term * np.exp(-norm**2 / (4*eta**2))
#         Cfar_ij += term
#         
#     Cfar_ij = Cfar_ij * (4*np.pi / lattice.unitCellVol)
#     print(Cfar_ij)
#     Cfar_ij = Cfar_ij * np.exp( 1j * q @ Delta)
#         
#     return Cfar_ij
# 
# 
# def realSpaceSum(
#                   Delta,
#                   q, eta, sumdepth):
#     
#     RList = model.C_obj._buildList(lattice.latticeVectors, sumdepth)
#     q = np.array(q)
#     Cnear_ij = np.zeros([3,3] , dtype='complex128')
#     DeltaRList = [R-Delta for R in RList]
#         
#     for dR in DeltaRList:
#         norm = la.norm(dR)
#         y = eta*norm
#         t1 = np.outer(dR, dR) / norm**5
#         t1 = t1 * (3*erfc(y)  +  1/np.sqrt(np.pi) * (6*y + 4*y**3)*np.exp(-y**2))
#         t2 = np.eye(3) / norm**3
#         t2 = t2 * ( erfc(y) + 2*y * np.exp(-y**2) / np.sqrt(np.pi) )
#         term = t1 - t2
#         term = term * np.exp(1j * q @ (dR + Delta))
#         Cnear_ij += term
#         
#     Cnear_ij = Cnear_ij * np.exp(1j * q @ Delta)
#         
#     return Cnear_ij
# 
# 
# for sumdepth in range(1, 8):
#     i =  qSpaceSum(Delta, q, eta, sumdepth)
#     j = qSpaceSum(Delta, q, eta, sumdepth+1)
#     print(np.round(j-i, 10))
# 
# 
# =============================================================================

