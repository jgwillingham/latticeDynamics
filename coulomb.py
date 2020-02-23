# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:00:26 2020

@author: George Willingham
"""

import numpy as np
import scipy.linalg as la
from scipy.special import erfc 



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
                
                blocks[i].append( C_ij )
                if i == j:
                    # include self term
                    blocks[i][j] += self._Cself(i)
                    
                if la.norm(q) != 0:
                    # non-analytic term excluded where it is singular
                    norm = la.norm(q)
                    G0_term = np.outer(q, q) / norm**2
                    G0_term = G0_term * np.exp(- norm**2/(4*self.eta**2))
                    blocks[i][j] += (4*np.pi/self.lattice.volume) * G0_term
  
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
            term = term * np.exp(-1j * G @ Delta) 
            term = term * np.exp(-norm**2 / (4*self.eta**2))
            Cfar_ij += term
        
        Cfar_ij = Cfar_ij * (4*np.pi / self.lattice.volume)
        Cfar_ij = Cfar_ij * np.exp(1j * q @ Delta) 
        
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
        DeltaRList = [R+Delta for R in self.RList]
        
        for dR in DeltaRList:
            norm = la.norm(dR)
            y = self.eta*norm
            t1 = np.outer(dR, dR) / norm**5
            t1 = t1 * (3*erfc(y)  +  1/np.sqrt(np.pi) * (6*y + 4*y**3)*np.exp(-y**2))
            t2 = np.eye(3) / norm**3
            t2 = t2 * ( erfc(y) + 2*y * np.exp(-y**2) / np.sqrt(np.pi) )
            term = t1 - t2
            term = term * np.exp(1j * q @ (dR - Delta))
            Cnear_ij += term
        
        Cnear_ij = Cnear_ij * np.exp(1j * q @ Delta)
        
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