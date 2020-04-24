# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:34:12 2020

@author: George Willingham
"""

import numpy as np
import scipy.linalg as la
import matplotlib as mpl



class GreensFunction:
    
    
    def __init__(self, dynamicalMatrix):
        self.D = dynamicalMatrix
        
        
        
    def _blockSplit(self,
                    matrix, 
                    blockSize,
                    tol=10**-9):
        """
        Splits a given matrix into blocks of a given size

        Parameters
        ----------
        matrix : numpy matrix
                 The matrix to be split into blocks
        blockSize : int
                    dimension of the blocks
        tol : float, optional
              tolerance for checking if blockSize divides matrix size. 
              The default is 10**-9.

        Raises
        ------
        ValueError
            If matrix cannot be split evenly into blocks of size blockSize

        Returns
        -------
        blockList : list
            list of blocks
        """
        
        matrixSize = len(matrix)
        numBlocks_ = matrixSize/blockSize
        if numBlocks_ - np.round(numBlocks_) > tol:
            raise ValueError(f'Invalid block size. Matrix has size {matrixSize}')
        
        blockList = []
        blockRange = range(0, matrixSize, blockSize)
        for i in blockRange:
            row = []
            for j in blockRange:
                row.append(matrix[i:i+blockSize, j:j+blockSize])
            blockList.append(row)
        return blockList
            



    def _checkBlockTridiagonal(self,
                               blockList,
                               tol=10**-9):
        """
        Checks if given list of blocks is block tridiagonal

        Parameters
        ----------
        blockList : list
                    list of matrix blocks to test.
        tol : float, optional
              Tolerance for zero blocks. 
              The default is 10**-9.

        Returns
        -------
        bool
            returns whether the given blocks are tridiagonal.

        """
        
        absAverages = [[abs(block).mean() for block in row] 
                           for row in blockList]
        absAverages = np.array(absAverages)
        
        # check first two rows
        for i in range(2):
            if np.all(absAverages[i][i+2:] < tol):
                continue
            else:
                return False
        # check all remaining rows
        for i in range(2, len(absAverages)):
            if np.all(absAverages[i][:i-2] < tol) and np.all(absAverages[i][i+2:] < tol):
                continue
            else:
                return False
        
        return True



    def blockTridiag(self, 
                     matrix, 
                     maxBlockSize=60):
        """
        Block tridiagonalizes a given matrix. Returns the list of blocks

        Parameters
        ----------
        matrix : numpy matrix
                 matrix to be block tridiagonalized.
        maxBlockSize : int, optional
                       Maximum size of blocks. 
                       The default is 60.

        Returns
        -------
        blockList : list
            list of blocks of tridiagonal matrix.
        """
                
        for blockSize in range(3, maxBlockSize, 3):
            blockList = self._blockSplit(matrix, blockSize)
            isTridiagonal = self._checkBlockTridiagonal(blockList)
            if isTridiagonal:
                break
            
        return blockList
    
        
    def _g(self, 
          w, 
          eta, 
          E):
        """
        Returns an intermediate step Green's function used in the iterations

        Parameters
        ----------
        w : float
            Frequency.
        eta : float
            Frequency imaginary part.
        E : ndarray
            matrix used in iterations.

        Returns
        -------
        gi : ndarray
            intermediate step Green's fucntion
        """
        
        m = np.eye(len(E))*(w**2 + 1j*eta) - E
        gi = la.inv(m)
        
        return gi
    
    
    
    def _iterate(self,
                w, 
                eta, 
                a_prev, 
                b_prev, 
                Es_prev, 
                E_prev):
        """
        Carries out the iteration procedure. For more info on the algorithm,
        see the following paper:
            
        M P Lopez Sancho et al 1985 J. Phys. F: Met. Phys. 15 851

        Parameters
        ----------
        w : float
            frequency.
        eta : float
              frequency imaginary part.
        a_prev : ndarray
                 alpha as defined in Sancho paper.
        b_prev : ndarray
                 beta as defined in Sancho paper.
        Es_prev : ndarray
                  Es as defined in Sancho paper.
        E_prev : ndarray
                 E as defined in Sancho paper.

        Returns
        -------
        a_new : ndarray
                New alpha value.
        b_new : ndarray
                New beta value.
        Es_new : ndarray
                 New Es value.
        E_new : ndarray
                New E value.

        """
        
        g_prev = self._g(w, eta, E_prev)
        
        a_new  = a_prev @ g_prev @ a_prev
        b_new  = b_prev @ g_prev @ b_prev
        Es_new = Es_prev + a_prev @ g_prev @ b_prev
        E_new  = E_prev + a_prev @ g_prev @ b_prev + b_prev @ g_prev @ a_prev
        
        return a_new, b_new, Es_new, E_new
    
    
    
    def greensFunc(self,
                   w,
                   eta, 
                   a, 
                   b,
                   Es,
                   E,
                   iterNum=35):
        """
        Calculates the surface Green's function at frequency w (TeraRad/s)

        Parameters
        ----------
        w : float
            frequency.
        eta : float
              frequency imaginary part.
        a, b, Es, E : ndarray
                      Initial iteration parameters.

        iterNum : int, optional
                  Number of iterations. The default is 35.
                  Checking convergence with the convergenceTest method is 
                  recommended.

        Returns
        -------
        G : ndarray
            Surface Green's function.

        """
        
        counter = 0
        while counter <= iterNum:
            a, b, Es, E = self._iterate(w, eta, a, b, Es, E)
            counter += 1
            
        G = la.inv( np.eye(len(Es))*(w**2 + 1j*eta) - Es )
        
        return G
    
    
    
    def LDOS(self,
             w, 
             blocks, 
             eta=10**-4, 
             iterNum=35):
        """
        Calculate the local density of states (LDOS) at the surface at a
        given frequency w and for a given dynamical matrix (evaluated at some
        wavevector q).

        Parameters
        ----------
        w : float
            frequency.
        blocks : list
                 List of blocks from the block tridiagonalization.
        eta : float, optional
              Frequency imaginary part. The default is 10**-4.
        iterNum : int, optional
                  Number of iterations. The default is 35.

        Returns
        -------
        A : float
            Local density of states.
        """
        
        a = blocks[0][1]
        b = blocks[1][0]
        Es = blocks[0][0]
        E = blocks[1][1]
        
        G_w = self.greensFunc(w, eta, a, b, Es, E, iterNum)
        
        A = (-1/np.pi)*np.imag(G_w.trace() )
        
        return A
    
    
    
    def convergenceTest(self,
                        w, 
                        eta, 
                        blocks,
                        iterMax=50):
        """
        Tests the convergence of the algorithm for a given parameter set

        Parameters
        ----------
        w : float
            frequency.
        eta : float
              frequency imaginary part.
        blocks : list
                 List of blocks from block tridiagonalization.
        iterMax : int, optional
                  Maximum number of iterations to run. 
                  The default is 50.

        Returns
        -------
        conv : list
               List of changes in Es for each iteration.
        Also the number of iterations to convergence is printed.
        """
        
        a = blocks[0][1]
        b = blocks[1][0]
        Es = blocks[0][0]
        E = blocks[1][1]
        
        counter = 0
        conv = [Es]
        while counter <= iterMax:
            Es_prev = Es
            a, b, Es, E = self._iterate(w, eta, a, b, Es, E)
            conv.append(Es - Es_prev)
            
            if np.all(Es-Es_prev == np.zeros(len(a))):
                print(f'Converges after {counter} iterations')
                return None
            counter += 1
            
        print(f'Did not converge after {iterMax} iterations')
        
        return conv
    
    
    
    def spectralFunction(self,
                          D,
                          qList, 
                          fList, 
                          eta=10**-4, 
                          iterNum=35,
                          showProgress=True):
        """
        Calculates the full spectral function given a callable dynamical
        matrix D, a list of frequency values (in THz) and a list of 
        wavevectors.

        Parameters
        ----------
        D : function
            Function returning the dynamical matrix at a given k-value.
        qList : list
                List of wavevectors to calculate LDOS over.
        fList : list
                List of frequencies (in THz) to calculate LDOS for 
                at each wavevector.
        eta : float, optional
              frequency imaginary part. 
              The default is 10**-4.
        iterNum : int, optional
                  Number of iterations. 
                  The default is 35.
        showProgress : bool, optional
                       To print progress updates or not. 
                       The default is True.

        Returns
        -------
        ndarray
            Array containing a list of LDOS values for each q point. The list
            has a value for each f in fList.

        """

        # convert from THz
        wList = [2*np.pi*f for f in fList]
        
        A_kw = []
        progress=1
        for q in qList:
            if showProgress:
                print(f'\r{progress}/{len(qList)}', end='')
            dynamical_matrix = D(q)
            blocks = self.blockTridiag(dynamical_matrix)
            energy_curve = [self.LDOS(w, blocks, eta, iterNum) 
                                for w in wList]
            A_kw.append(energy_curve)
            progress += 1
        
        if showProgress:
            print('\nDone!')
        
        return np.array(A_kw)
    
    
    
    def plot(self):
        pass
    
    
    
    