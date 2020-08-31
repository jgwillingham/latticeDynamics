# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:34:12 2020

@author: George Willingham
"""

import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


# This is used for a warning . It is just for limiting the warning output to
# just a given string instead of extra information
def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning





class GreensFunction:
    """
    Class containing all necessary methods for surface Green's functions 
    calculations. The algorithm is the iterative approach described by 
    Sancho et al. in the paper
    
    M P Lopez Sancho et al 1985 J. Phys. F: Met. Phys. 15 851
    
    Parameters
    ----------
    dynamicalMatrix : function
                      A callable SLAB dynamical matrix. 
                      
    NOTE : Only part of the slab dynamical matrix is used so it is best to 
    keep the given slab size as small as possible to maximuze performance.
    The slab only needs to be large enough to capture the surface principal 
    layer and a single bulk principal layer.
                      
    """
    
    def __init__(self, 
                 dynamicalMatrix):
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



    def _blockTridiag(self, 
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
            try:
                blockList = self._blockSplit(matrix, blockSize)
            except ValueError:
                continue
            isTridiagonal = self._checkBlockTridiagonal(blockList)
            if isTridiagonal:
                break
        if blockSize > len(matrix)/3:
            warnings.warn('(!) Slab surface principal layers are interacting. '\
                          'A larger slab is recommended to '\
                          'accurately capture bulk principal layers.', Warning)
            
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
        a_prev, b_prev, Es_prev, E_prev : ndarrays
                                          Iterated parameters as defined in 
                                          the Sancho et al. paper.

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
        # perhaps the trace should be truncated to view different surface depths?
        
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
    
    
    def _getPLBlockSize(self,
                        qTest=np.array((0.01, 0.01, 0.01))):
        """
        Get principal layer block size in dynamical matrix

        Parameters
        ----------
        qTest : ndarray, optional
                Test wavevector to compute dynamical matrix and block
                tridiagonalize. 
                The default is np.array((0.01, 0.01, 0.01)).

        Returns
        -------
        int
            Principal layer block size.
        """
        
        dynamicalMatrix = self.D( qTest , 10**9) # 10**9 is just a big number placeholder. 
        blockList = self._blockTridiag(dynamicalMatrix)        
        self.PLBlockSize = len(blockList[0][0])
        
        return self.PLBlockSize
        
    
    
    
    def spectralFunction(self,
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
        
        if not hasattr(self, 'PLBlockSize'):
            self.PLBlockSize = self._getPLBlockSize()
            
        n = int(2*self.PLBlockSize)/3 # to get just surface PL and first bulk PL 
        
        A_qw = []
        progress=1
        
        for q in qList:
            if showProgress:
                print(f'\r{progress}/{len(qList)}', end='')
            dynamicalMatrix_corner = self.D(q, n) # just the necessary blocks
            blocks = self._blockSplit(dynamicalMatrix_corner, self.PLBlockSize)
            energy_curve = [self.LDOS(w, blocks, eta, iterNum) 
                                for w in wList]
            A_qw.append(energy_curve)
            progress += 1
        
        if showProgress:
            print('\nDone!')
        
        return np.array(A_qw)
    
    
        
    def plotLDOS(self,
                 A_qw,
                 qPathParts,
                 fList,
                 qLabels=[],
                 title='LDOS',
                 cmap='hot',
                 markercolor='w',
                 figsize=(12,8),
                 numYLabels=5):
        """
        Plot a surface spectral function (or LDOS) as calculated by the 
        spectralFunction method.

        Parameters
        ----------
        A_qw : ndarray
               Array containing the calculated LDOS. 
               For the i^th wavevector in the BZ path , A_kw[i] should be a 
               1D array of DOS values for each energy.
        qPathParts : list
                     List of separate paths in reicprocal space. This is given
                     by the buildPath method in the Model class
        fList : list
                List of frequencies in THz.
        qLabels : list, optional
                  List of strings for names of high symmetry points along 
                  calculated path. 
                  The default is [].
        title : str, optional
                Title of the plot. 
                The default is 'LDOS'.
        cmap : str, optional
               Name of matplotlib colormap. 
               The default is 'hot'.
        markercolor : str, optional
                      Color of marker lines along path. 
                      The default is 'w'.
        figsize : tuple, optional
                  Figure size. 
                  The default is (12,8).
        numYLabels : int, optional
                     Number of labels on the frequency (Y) axis other than 0. 
                     The default is 5.

        Returns
        -------
        Displays plot

        """
        
        A = A_qw.T[::-1, :] # transpose and reverse order of energy lists
                            # ^this is needed for imshow plotting
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(A, 
                  aspect='auto',
                  norm=mpl.colors.LogNorm(A.min(), A.max()),
                  cmap=cmap)
        
        pathLens = [len(path) for path in qPathParts]
        cumPathLens = [sum(pathLens[:i])-i for i in range(len(pathLens)+1)]
        
        ax.set_xticks( cumPathLens )
        ax.set_xticklabels(qLabels)
        ax.set_yticks( [0] + [len(fList)*((n+1)/numYLabels) 
                                 for n in range(numYLabels)])
        ax.set_yticklabels([np.round(max(fList)*(1-x) + min(fList)*x ,2) 
                                for x in np.linspace(0, 1, numYLabels+1)])
        
        ax.set_ylabel('$\\nu$ (THz)', rotation=0, labelpad=40)
        ax.set_title(title)
        
        for pathPart in cumPathLens[1:-1]:
            ax.axvline(pathPart, color=markercolor, alpha=0.5)
            
        plt.show()
            
        
        
    def isofreq(self,
                  w,
                  qxPath, 
                  qyPath,
                  showProgress=True):
        """
        Calculates the LDOS on an isofrequency surface over a parallelogram 
        region of the surface Brillouin zone.

        Parameters
        ----------
        D : function
            Callable slab dynamical matrix.
        w : float
            frequency to calculate isoenergy (isofrequency) surface.
        qxPath : ndarray
                 1D array of wavevectors along bottom edge of parallelogram 
                 region
        qyPath : ndarray
                 1D array of wavevectors along edge of parallelogram which is
                 not parallel to qxPath values
                 
        Returns
        -------
        ndarray
            2D array containing the isoenergy surface values over the desired
            region.
        """

        wList = [w]
        data = []
        progress = 1
        for q in qxPath:
            if showProgress:
                print(f'\rCut {progress} of {len(qxPath)}', end='')
            q = np.array(q)
            ycut = np.array([np.array(qy) + q for qy in qyPath])
            A_qx = self.spectralFunction(ycut, wList, showProgress=False)
            data.append(A_qx)
            progress += 1
            
        if showProgress:
            print('\nDone!')
        
        return np.array(data)
    
    
    
    def plotIsofreq(self,
                      isofreq,
                      cmap='hot',
                      figsize=(10, 8),
                      xticks=[],
                      yticks=[],
                      xtickLabels=[],
                      ytickLabels=[]):
        """
        Plot isofrequency calculation.
        
        NOTE : even non-rectangular regions will be plotted as rectangles. 
        This will hopefully be changed in a later version.

        Parameters
        ----------
        isofreq : ndarray
                  Array containing the results of an isofrequency calculation.
        cmap : str, optional
               Matplotlib colormap.
               The default is 'hot'.
        figsize : tuple, optional
                  Figure size. 
                  The default is (10, 8).

        Returns
        -------
        Displays plot

        """
        
        iso = isofreq[:,:,0].T[::-1, :]
        
        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(iso,
                  norm=mpl.colors.LogNorm(iso.min(), iso.max()),
                  cmap=cmap)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtickLabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytickLabels)
        plt.show()
        
    
    