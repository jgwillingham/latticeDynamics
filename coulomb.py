# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:00:26 2020

@author: George Willingham
"""

import numpy as np
import scipy.linalg as la
from scipy.special import erfc, gamma, gammaincc
from latticeDynamics.lattice import Lattice
from latticeDynamics.slab import Slab



class Coulomb:
    """
    Class containing all methods needed for calculating the Coulomb
    contribution to the dynamical matrix via the Ewald method.
        
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
        self.GSumDepth = GSumDepth
        self.RSumDepth = RSumDepth
        
        if isinstance(lattice, Lattice):
            self.dim = 3 # dimension of Ewald sum
            self._latticeVectors = lattice.lattice_vectors
            self._reciprocalVectors = lattice.reciprocal_vectors
            self._cellVol = lattice.volume
            self._unitCell = lattice._unitCell
            self._atomsPerCell = lattice.atomsPerUnitCell
            self.charges = np.array(charges)
            self.ewald = self._bulkEwald
            
        elif isinstance(lattice, Slab):
            self.dim = 2
            self._latticeVectors = lattice.meshPrimitives
            self._reciprocalVectors = lattice.meshReciprocals
            self._cellVol = lattice.area
            self._unitCell = lattice.slabCell
            self._atomsPerCell = lattice.atomsPerSlabCell
            self.charges = np.array([charges for cell in range(lattice.numCells)]).flatten()
            self.ewald = self._slabEwald
            
            
        if eta == 'default':
            self.eta = 4*(self._cellVol)**(-1/self.dim)
        else:
            self.eta = eta
        self.GList = self._buildList(self._reciprocalVectors,
                                     GSumDepth)
        self.RList = self._buildList(self._latticeVectors,
                                     RSumDepth)
        self.Z = self.getChargeMatrix()




    def C(self, 
          q,
          num_blocks=None):
        """
        Get full block Coulomb matrix at wavevector q

        Parameters
        ----------
        q : array_like
            Wavevector where matrix should be calculated
        num_blocks : int
            Number of atom 3x3 blocks to construct 

        Returns
        -------
        _C : numpy matrix
             Full num_block block Coulomb matrix at wavevector q 
        """

        q = np.array(q)
        if num_blocks == None:
            n = self._atomsPerCell
        else:
            n = num_blocks
        blocks = []
        
        for i in range(n):
            blocks.append([])
            xi = self._unitCell[i].coords_cartesian
            
            for j in range(n):
                xj = self._unitCell[j].coords_cartesian
                intracell_distance = xj - xi
                
                C_ij = self.ewald(q, intracell_distance)
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
        Ci_self = np.zeros([3, 3], dtype='complex128')
        xi = self._unitCell[i].coords_cartesian
        
        for j in range(self._atomsPerCell):
            Zfactor = self.charges[j] / self.charges[i]
            xj = self._unitCell[j].coords_cartesian
            intracell_distance = xj - xi
            
            C_ij = self.ewald(Gamma, intracell_distance)
            
            Ci_self -= Zfactor*C_ij
        
        return Ci_self
    
    
    
    
    def _bulkEwald(self,
                   q,
                   intracell_distance):
        """
        Given an intracell distance between two atoms , this calculates 
        the Ewald summation for the bulk crystal at wavevector `q`. 
        It returns the block of the Coulomb contribution to the dynamical
        matrix relating the two atoms separated by `intracell_distance`

        Parameters
        ----------
        q : arraylike
            Wavevector to calculate at.
        intracell_distance : arraylike
            Intracell distance between two atoms.

        Returns
        -------
        C_ij : matrix
            Block i,j of Coulomb contribution to dynamical matrix.

        """
        
        C_far = self._qSpaceSum(q, 
                                intracell_distance, 
                                intracell_distance)
        C_near = self._realSpaceSum(q, 
                                    intracell_distance, 
                                    intracell_distance)
        C_ij = C_far + C_near
        
        return C_ij
    
    
    
    def _slabEwald(self,
                   q,
                   intracell_distance):
        """
        Given an intracell distance between two atoms , this calculates 
        the Ewald summation for the slab at wavevector `q`. 
        It returns the block of the Coulomb contribution to the dynamical
        matrix relating the two atoms separated by `intracell_distance`

        Parameters
        ----------
        q : arraylike
            Wavevector to calculate at.
        intracell_distance : arraylike
            Intracell distance between two atoms.

        Returns
        -------
        C_ij : matrix
            Block i,j of Coulomb contribution to dynamical matrix.
        """
        Delta_parallel, Delta_normal = self.lattice.projectVector(intracell_distance)
        
        if la.norm(Delta_normal) > 10**-7:
            C_ij = self._differentPlaneSum(q, 
                                           Delta_parallel, 
                                           Delta_normal)
            
        else:
            C_far = self._qSpaceSum(q, 
                                    Delta_parallel, 
                                    intracell_distance)
            C_near = self._realSpaceSum(q, 
                                        Delta_parallel, 
                                        intracell_distance)
            C_ij = C_far + C_near 
            # C_ij = C_ij + np.eye(3)*4/(3*np.sqrt(np.pi)) # from self-interaction
        
        
        return C_ij
            
          
            
                 
    def _qSpaceSum(self,
                   q,
                   Delta,
                   intracell_distance):
        """
        Reciprocal lattice sum in d-dimensional Ewald summation

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
        d = self.dim
        Delta = np.array(Delta)

        Cfar_ij = np.zeros([3,3], dtype='complex128')
        QGList = [np.array(q+G) for G in self.GList]
        
        if la.norm(q) != 0: # include G=0 term when non-singular
            QGList.append(q)

        for G in QGList:
            norm = la.norm(G)
            term = np.outer(G, G) / norm**(d-1)
            term = term * np.exp(-1j * G @ intracell_distance) 
            alpha = (d-1)/2
            x = norm / (2*self.eta)
            term *= gammaincc(alpha, x**2) * gamma(alpha)
            Cfar_ij += term
        
        Cfar_ij = Cfar_ij * (2*np.sqrt(np.pi))**(d-1) / self._cellVol
        Cfar_ij = Cfar_ij * np.exp(1j * q @ Delta) 
        
        
        return Cfar_ij
    
    
    
    def _realSpaceSum(self,
                      q,
                      Delta,
                      intracell_distance):
        """
        Direct lattice sum in d-dimensional Ewald summation

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
        DeltaRList = [R + intracell_distance for R in self.RList]
        
        for dR in DeltaRList:
            norm = la.norm(dR)
            y = self.eta*norm
            t1 = np.outer(dR, dR) / norm**5
            t1 = t1 * (3*erfc(y)  +  1/np.sqrt(np.pi) * (6*y + 4*y**3)*np.exp(-y**2))
            t2 = np.eye(3) / norm**3
            t2 = t2 * ( erfc(y) + 2*y * np.exp(-y**2) / np.sqrt(np.pi) )
            term = t1 - t2
            term = term * np.exp(1j * q @ (dR - intracell_distance))
            Cnear_ij += term
        
        Cnear_ij = Cnear_ij * np.exp(1j * q @ Delta)
        
        return -1*Cnear_ij
    
    
    
    def _differentPlaneSum(self,
                            q,
                            Delta_parallel,
                            Delta_normal):
        """
        Implements part of the slab geometry Ewald summation when the origin 
        is not in the plane

        Parameters
        ----------
        q : arraylike
            wavevector.
        Delta_parallel : arraylike
            component of inter-atomic distance parallel to the slab surfaces.
        Delta_normal : arraylike
            component of inter-atomic distance normal to the slab surfaces.

        Returns
        -------
        C_ij : matrix
            block of coulomb contribution to dynamical matrix.

        """
        
        C_ij = np.zeros([3,3], dtype='complex128')
        
        qGList = [q + G for G in self.GList]
        
        if la.norm(q) != 0:
            qGList.append(q) # include G=0 term when non-singular
            
        for qG in qGList:
            qGnorm = la.norm(qG)
            Delta_norm = la.norm(Delta_normal)
            
            term1 = np.outer(qG, qG) / qGnorm
            
            term2 = 1j*np.outer(qG, Delta_normal) / Delta_norm
            term2 += 1j*np.outer(Delta_normal, qG) / Delta_norm

            term3 = qGnorm*np.outer(Delta_normal, Delta_normal) / Delta_norm**2

            full_term = (term1 - term2 - term3)*np.exp(-1j*qG @ Delta_parallel)
            full_term = full_term * np.exp( - Delta_norm * qGnorm)
            
            C_ij = C_ij + full_term
        
        C_ij = C_ij * (2*np.pi/self._cellVol)*np.exp(1j * q @ Delta_parallel)
        
        return C_ij
    
    
        
        
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
        
        sumRange = range(-sumDepth, sumDepth+1)
        vectors = list(vectors)
        if self.dim == 3: 
            zSumRange = sumRange
        elif self.dim == 2:
            zSumRange = [0] # only sum over third vector if in 3D bulk
            vectors.append(np.zeros(3))
            
        v = vectors
        
        # make list of reciprocal/direct lattice vectors to sum over
        Vec = lambda n1,n2,n3 : n1*v[0] + n2*v[1] + n3*v[2]
        
        List = []
        for n1 in sumRange:
            for n2 in sumRange:
                for n3 in zSumRange:
                    if n1==n2==n3==0:
                        pass
                    else:
                        vector = Vec(n1, n2, n3)
                        List.append(vector)
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
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
