# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:49:09 2020

@author: George Willingham
"""


import numpy as np
import scipy.linalg as la



class RigidIon:
    """
    Class containing all functions needed for the short-range force
    contributions to the dynamical matrix
    
    Parameters
    ----------
    lattice : Lattice or Slab object
              An instance of the Lattice/Slab class for the desired crystal
              
    couplingArray : 2D array_like
                    2D array containing the force constants for calculating 
                    the force constant matrices.
                    Element [i,j] should contain a list of the couplings
                    for interactions between atom_i and atom_j as listed in 
                    the Lattice object.
                    
                    e.g. couplingArray[i, j] = (A_ij , B_ij)
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
        try:
            self.atomsPerUnitCell = self.lattice.atomsPerUnitCell # if given Lattice object
            self._cellVolume = self.lattice.volume
        except AttributeError:
            self.atomsPerUnitCell = self.lattice.atomsPerSlabCell # if given Slab object
            self._cellVolume = self.lattice.bulk.volume
    
    
    
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
        A *= (e**2 / abs(2*self._cellVolume))
        B *= (e**2 / abs(2*self._cellVolume))


        Phi = (A-B)*np.outer(bond_ij, bond_ij) / (la.norm(bond_ij)**2)
        Phi += B*np.eye(3)

        #for x_i in range(3):
        #    for x_j in range(3):
        #        Phi[x_i, x_j] = (A - B)*bond_ij[x_i]*bond_ij[x_j] / (la.norm(bond_ij)**2)
        #        if x_i == x_j:
        #            Phi[x_i, x_j] += B
                
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
                bond_ij = neighbor[1][0] # neighbor is a pair ( (label_i, label_j), [bond_ij, latVec] )
                latVec = neighbor[1][1]
                Phi_ij = self._forceConstantMatrix(bond_ij, A, B)
                R_ij = R_ij + Phi_ij*np.exp( 1j *q @ latVec )
                
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
          q,
          num_blocks=10**9):
        """
        Get the full rigid ion matrix at wavevector q

        Parameters
        ----------
        q : array_like
            Wavevector where matrix should be calculated
        num_blocks : int
            number of atom 3x3 blocks to calculate (used for Green's functions)
            Default is 10**9 (big enough to always capture all blocks)
        Returns
        -------
        _R : numpy matrix
             Full block matrix contribution to dynamical matrix from 
             rigid ions.
        """
        if not isinstance(q, np.ndarray): q = np.array(q)
        if num_blocks == 10**9:
            n = self.atomsPerUnitCell
        else:
            n = num_blocks
        blocks = []
        
        for i in range(n):
            blocks.append([])
            for j in range(n):
                blocks[i].append( self._Rblock(i, j, q) )
                if i == j:
                    blocks[i][j] += self._Rself(i)
                    
                
        _R = np.matrix(np.block( blocks ))

        return _R