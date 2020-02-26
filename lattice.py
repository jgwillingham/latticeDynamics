# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:40:56 2020

@author: George Willingham
"""


import numpy as np
import scipy.linalg as la
from crystals import Atom, Crystal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




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
        
        
        
# %%
        
        
class Slab:
    """
    For slab calculations. Right now just cubic lattices.
    """
    
    def __init__(self, 
                 lattice, 
                 surface, 
                 numCells):
        self.lattice = lattice
        self.bulkUnitCell = lattice._unitCell
        self.surface = surface
        self.numCells = numCells
        self.hkl = self._handleMillerIndices(surface)
        self.surfaceNormal, self.a_normal = self._getNormalVectors()
    
    
    def _getNormalVectors(self):
        
        (h,k,l) = self.hkl
        (b1, b2, b3) = self.lattice.reciprocal_vectors
        G_hkl = h*b1 + l*b2 + k*b3
        surfaceNormal = G_hkl / la.norm(G_hkl) 
        # this^ is the unit vector in direction normal to surface
        spatialPeriod = (2*np.pi) / la.norm(G_hkl) # like T = 2 pi/w
        self.d_hkl = spatialPeriod
        # this^ is the distance between hkl lattice planes
        a_normal = (spatialPeriod)*surfaceNormal
        # this^ is the vector connecting lattice planes
        
        return surfaceNormal, a_normal
    
    
    def projectVector(self, latticeVector):
        
        R = latticeVector
        n = self.surfaceNormal
        R_normal = (R @ n)*n # projects latticeVector onto surface normal
        R_inplane = R - R_normal # remaining component parallel to surface
        
        components = [np.round(comp, 9) for comp in [R_inplane, R_normal]]
        
        return components
    
    
    def _handleMillerIndices(self, miller_indices):
                
        hkl = []
        sign = 1
        for char in miller_indices:
            if char == '-':
                sign = -1
            else:
                inx = int(char)
                hkl.append(sign*inx)
                sign = 1
    
        return hkl
            
            
    
    def _getLatticePlanes(self):
        """
        Looks for non-similar lattice planes in the given direction.
        Projects all atom positions in the unit cell onto the surface normal
        along with all lattice vectors looking for distinct projections.
        
        """
        
        distinct_planes = set() # use a set so all elements are distinct
        projections = []
        
        for atom in self.lattice.atoms:
            r_inplane, r_normal = self.projectVector(atom.coords_cartesian)
            projections.append((r_inplane, r_normal))
            r_normal = tuple(np.round(r_normal, 9)) # chops near 0 values
            distinct_planes.add(r_normal)
            
        for latvec in self.lattice.lattice_vectors:
            R_inplane, R_normal = self.projectVector(latvec)
            projections.append((R_inplane, R_normal))
            R_normal = tuple(np.round(R_normal, 9)) # chops near 0 values
            distinct_planes.add(R_normal)
            
        return projections, list(distinct_planes)
    
    
            
        
    def _getPlaneBasis(self, projection, searchWidth=1):
        """
        First attempt at algorithm for finding a primitive lattice plane basis.
        Need to check more cases
        """        
        searchRange = range(-searchWidth, searchWidth+1)
        (a1, a2, a3) = self.lattice.lattice_vectors
        planarAtoms = [] # list of nearby atom positions that are in the same plane
        atomPosition = projection[0] + projection[1]
        
        # get a list of coplanar atom positions
        for n1 in searchRange:
            for n2 in searchRange:
                for n3 in searchRange:
                    Rl = n1*a1 + n2*a2 + n3*a3
                    nomineePosition = atomPosition + Rl
                    r_inplane, r_normal = self.projectVector(nomineePosition)
                    if all(r_normal == projection[1]) and la.norm(r_inplane)!=0:
                        planarAtoms.append(r_inplane)
        
        angle = lambda v1, v2: np.arccos((v1@v2)/(la.norm(v1)*la.norm(v2))) # gets angle between two vectors
        vectorLengths = [la.norm(vector) for vector in planarAtoms]
        nearest_inx = np.argsort(vectorLengths) # list of indices for planarAtoms in order of closest to farthest
        plane_basis = [ planarAtoms[nearest_inx[0]] ]
        
        for i in nearest_inx:
            vector = planarAtoms[i]
            if all([angle(vector, basisVector)% np.pi > 10**-5 
                            for basisVector in plane_basis]): # check if vector is not parallel to other basis vector
                plane_basis.append(vector)
                break
        
        return plane_basis
            
            
     
    
            
        
# %%
        
# =============================================================================
# unitCell = [('Cd', np.array((0, 0, 0))),
#             ('Te', np.array((1/4, 1/4, 1/4))),
#             ('Cd', np.array((0, 1/2, 1/2))),
#             ('Cd', np.array((1/2, 0, 1/2))),
#             ('Cd' , np.array((1/2, 1/2, 0))),
#             ('Te', np.array((3/4, 3/4, 1/4))),
#             ('Te', np.array((1/4, 3/4, 3/4))),
#             ('Te' , np.array((3/4, 1/4, 3/4)))]
# latvecs = [(1, 0, 0) , (0, 1, 0), (0,0,1)]
# =============================================================================
latvecs = [(1/2, 1/2, 0) , (1/2, 0, 1/2) , (0, 1/2, 1/2)]
#latvecs = [(1, 0, 0) , (0, 1, 0), (0,0,1)]
unitCell = [('Cd', np.array((0, 0, 0))),
          ('Te', np.array((1/4, 1/4, 1/4)))]
a = 6.6
latvecs = [a*np.array(v) for v in latvecs]
lattice = Lattice(unitCell, latvecs)
slab = Slab(lattice, '101', 1)    
projections, planes = slab._getLatticePlanes()    
basis = slab._getPlaneBasis(projections[2])
print(slab.a_normal)
print(planes)
print(basis)







