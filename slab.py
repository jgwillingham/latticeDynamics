# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:59:50 2020

@author: George Willingham
"""
import numpy as np
import scipy.linalg as la
from crystals import Atom, Crystal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class Slab(Crystal):
    """
    Given a surface (hkl), a Slab object constructs all the necessary geometry
    needed for slab phonon calculations.
    Inherits from the Crystal class.
    
    
    NOTE: The (hkl) surface parameter must be the h,k, and l coefficients for
    the PRIMITIVE reciprocal lattice vectors. This is in contrast to
    conventions for identifying surfaces in cubic crystals.
    
    NOTE: The term 'surface-adapted lattice vector' is used to refer to 
          the primitive lattice vectors contructed so that two are parallel
          to the surface and one is not.
    
    Parameters
    ----------
    bulkLattice : Lattice object
            A Lattice object for the bulk crystal structure
            
    surface : str
            A string containing the miller indices for the surface of the slab.
            e.g. '110' or '1-11'
    
    numCells : int
            The number of cells thick the slab should be
    
    searchWdith : int (optional)
                Determines how deep a search in all three dimensions should be
                made looking for a new set of primitive lattice vectors 
                adapted to the surface.
                Default is 1
    """
    
    def __init__(self, 
                 bulkLattice, 
                 surface, 
                 numCells,
                 searchWidth=1):
        self.bulk = bulkLattice
        self.surface = surface
        self.numCells = numCells
        self.hkl = self._handleMillerIndices(surface)
        self.surfaceNormal, self.planeSpacing = self._getNormalVectors()
        
        self.meshPrimitives = self._getMeshPrimitives(searchWidth)
        self.area = la.norm(np.cross(self.meshPrimitives[0], self.meshPrimitives[1])) # area of mesh unit cell
        self.zPrimitive = self._getOutOfPlanePrimitiveVector(searchWidth)
        self.adaptedLatticeVectors = [self.meshPrimitives[0], 
                                      self.meshPrimitives[1],
                                      self.zPrimitive]
        
        self.meshReciprocals = self._getMeshReciprocalVectors()
        
        self.slabCell, slabVectors = self._buildSlabCell()
        self.atomsPerSlabCell = len(self.slabCell)
        Crystal.__init__(self, self.slabCell, slabVectors) 
        self.atomicWeights = [atom.mass for atom in self.slabCell]
        self.atomLabels = [f'{inx}_{atom.element}' 
                      for atom, inx in 
                      zip(self.slabCell, range(self.atomsPerSlabCell))]
        
        
    def getNeighbors(self, 
                     threshold, 
                     cellSearchWidth=1):
        """
        Get lists of neighbors for each atom in the slab unit cell. This is
        the function that really makes the slab a slab. Any atom that does 
        not lie within the slab is not included in the neighbor list.

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
                    Dictionary containing every atom in the slab unit cell's 
                    neighbors as the key-value pairs: SYMBOL-NEIGHBOR_LIST.
                    Neighbor lists are accessed with element name given in
                    the unitCell parameter when Lattice object is instantiated.
                    e.g. 
        """
        
        searchWidth = range(-cellSearchWidth, cellSearchWidth+1)
        neighbors = {atomLabel :[] for atomLabel in self.atomLabels}
        sign = np.sign(self._cartToFrac(self.slabCell[-1].coords_cartesian)[2])
        # This^ is the sign of the largest fractional z-component in the slab.
        # It is used to determine whether a given position is within the slab 
        
        (a1, a2, a3) = self.lattice_vectors
        
        for atom_i, label_i in zip(self.slabCell , self.atomLabels):
            Ri = atom_i.coords_cartesian
            
            for atom_j, label_j in zip(self.slabCell , self.atomLabels):
                xj = atom_j.coords_cartesian
                
                for n1 in searchWidth:
                    for n2 in searchWidth:
                        #for n3 in searchWidth:

                            latVec = n1*a1 + n2*a2 #+ n3*a3
                            Rj = xj + latVec
                            bond_ij = Rj - Ri
                            distance_ij = la.norm(bond_ij)
                            # check if neighbor is close enough and not the same
                            if (distance_ij > 10**-9 and 
                                distance_ij <= threshold):
                                
                                frac_coords = np.round(self._cartToFrac(Rj),9)
                                
                                # check if neighbor is within slab
                                if (not abs(frac_coords[2]) > 1 and 
                                    np.sign(frac_coords[2]) != -1*sign):

                                    neighbors[label_i].append( 
                                        ( (label_i, label_j) , np.round(bond_ij,9) ) 
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
        color_id = np.linspace(0, 1, self.atomsPerSlabCell)
        
        atom_neighbors = self.neighbors[atomLabel]
        
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        
        ax.scatter((0), (0), (0) , s=atomSize, color='y')
        
        for atom, i in zip(self.atomLabels, color_id):
            neighbor_coords = [neighbor[1] for neighbor in atom_neighbors 
                               if neighbor[0][1]==atom]
            if len(neighbor_coords) != 0:
                x, y, z = zip(*neighbor_coords)
                ax.scatter( x, y, z, s=atomSize, color=cmap(i), label=atom)
        
        ax.set_axis_off()
        ax.legend()
        plt.show()
        
    
    
    def visualizeSlab(self,
                      length=3,
                      width=3,
                      atomSize=400):
        
        cmap = plt.cm.RdGy
        color_id = np.linspace(0, 1, self.bulk.atomsPerUnitCell)
        (a1, a2, a3) = self.adaptedLatticeVectors
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        
        for atom, i in zip(self.slabCell, color_id):
            for n1 in range(length):
                for n2 in range(width):
                    for n3 in range(self.numCells):
                                    
                        coords = atom.coords_fractional + n1*a1 + n2*a2 + n3*a3
                        x, y, z = coords
                        ax.scatter(x, y, z, s=atomSize, color=cmap(i))
    plt.show()
        
    
    
    def projectVector(self, 
                      latticeVector):
        """
        Decomposes a given vector into a component normal to the surface
        and a component parallel to the surface.

        Parameters
        ----------
        latticeVector : arraylike

        Returns
        -------
        components : list
                     A list containing 2 arrays: The in-plane component
                     and out-of-plane component of input vector
        """        
        R = latticeVector
        n = self.surfaceNormal
        R_normal = (R @ n)*n # projects latticeVector onto surface normal
        R_inplane = R - R_normal # remaining component parallel to surface
        components = [np.round(comp, 9) for comp in [R_inplane, R_normal]]
        
        return components
    
    
    
    def _getNormalVectors(self):
        """
        Get the unit surface normal and the lattice plane spacing.

        Returns
        -------
        surfaceNormal : array
                        Unit surface normal
        planeSpacing : array
                        plane spacing vector (has norm 1/|G_hkl|)
        """
        (h,k,l) = self.hkl
        (b1, b2, b3) = self.bulk.reciprocal_vectors
        G_hkl = h*b1 + l*b2 + k*b3
        surfaceNormal = G_hkl / la.norm(G_hkl) 
        # this^ is the unit vector in direction normal to surface
        spatialPeriod = (2*np.pi) / la.norm(G_hkl) # like T = 2 pi/w
        self.d_hkl = spatialPeriod
        # this^ is the distance between hkl lattice planes
        planeSpacing = (spatialPeriod)*surfaceNormal
        # this^ is the vector connecting lattice planes
        
        return surfaceNormal, planeSpacing

    
    
    def _handleMillerIndices(self, 
                             miller_indices):
        """
        Converts string of miller indices into integers.

        Parameters
        ----------
        miller_indices : str
                        string of miller indices
        Returns
        -------
        hkl : list
             list containing miller indices as ints
        """
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
        Get distinct lattice planes.

        Returns
        -------
        planeBases : dict
                    contains information on distinct lattice planes
        """
        projections = []
        for atom in self.bulk.atoms:
            r_inplane, r_normal = self.projectVector(atom.coords_cartesian)
            projections.append((r_inplane, r_normal))  
            
        for latvec in self.bulk.lattice_vectors:
            R_inplane, R_normal = self.projectVector(latvec)
            projections.append((R_inplane, R_normal))
            
        # convert numpy array to tuple so it can be included in set (hashable)
        planes = set(tuple(comps[1]) for comps in projections)
        set_of_latvecs = set(map(tuple, self.bulk.lattice_vectors))
        intersection = planes.intersection(set_of_latvecs)
        distinct_planes = planes.symmetric_difference(intersection)

        planeBases = {plane:[] for plane in distinct_planes}
        for plane in distinct_planes:
            planeArray = np.array(plane)
            for components in projections:
                if all(components[1]==planeArray):
                    planeBases[plane].append(components[0])
        
        return planeBases
    
    
    def _getOutOfPlanePrimitiveVector(self, 
                                      searchWidth=1):
        """
        Get the third primitive lattice vector for the crystal: the one
        out of the plane of the surface.

        Parameters
        ----------
        searchWidth : int (optional)
                      How many unit cells to search in all 3 dimensions
                      Default is 1.
        Returns
        -------
        zVector : array
                  Out-of-plane, surface-adapted lattice vector
        """
        searchRange = range(-searchWidth, searchWidth+1)
        (a1, a2, a3) = self.bulk.lattice_vectors
        n = self.surfaceNormal
        
        zVectors = []
        for n1 in searchRange:
            for n2 in searchRange:
                for n3 in searchRange:
                    Rl = n1*a1 + n2*a2 + n3*a3
                    if Rl @ n > 10**-5 and not n1==n2==n3==0:
                        zVectors.append(Rl)
        shortest_inx = np.argmin([la.norm(zVec) for zVec in zVectors])
        zVector = zVectors[shortest_inx]
        
        return zVector

    
            
    def _getMeshPrimitives(self, 
                           searchWidth=1):
        """
        Get primitive vectors for the 2D lattice planes (meshes) 
        in the desired direction.
        
        NOTE: lattice object must have primitive unit cell
        NOTE: a given plane may have all atoms displaced by some amount from
        the returned primitve vectors.
        
        
        Idea of Algorithm
        -----------------
        Since any two atomic positions in a Bravais lattice are connected
        by a lattice vector, the set of primitive vectors for the mesh
        must consist of lattice vectors which are parallel to the plane. 
        The algorithm here finds the shortest two distinct lattice vectors 
        parallel to the correct plane.
        
        Parameters
        ----------
        searchWdith : int (optional)
                      Determines how distant the search for adapted lattice
                      vectors should go
                      Default is 1.
                      
        Returns
        -------
        mesh_primitives : list
                          List containing two primitive lattice vectors 
                          which are parallel to the surface. These form a 
                          primitive set for the 2D lattice in the plane.
        """        
        searchRange = range(-searchWidth, searchWidth+1)
        (a1, a2, a3) = self.bulk.lattice_vectors
        n = self.surfaceNormal
        
        coplanars = [] # list of coplanar vectors
        for n1 in searchRange:
            for n2 in searchRange:
                for n3 in searchRange:
                    Rl = n1*a1 + n2*a2 + n3*a3
                    if abs(Rl @ n) <10**-10 and not n1==n2==n3==0:
                        coplanars.append(Rl)

        # order of indices for shortest to longest lattice vectors
        lengthOrder = np.argsort([la.norm(vector) 
                                   for vector in coplanars])
        # add shortest lattice vector to basis
        mesh_primitives = [ coplanars[lengthOrder[0]] ]

        # function that gets angle between two vectors
        # and test for non-parallel
        angle = lambda v1, v2: np.arccos((v1@v2)/(la.norm(v1)*la.norm(v2)))
        nonParallel = lambda  v1, v2 : angle(v1, v2)%np.pi > 10**-5
        
        for i in lengthOrder:
            vector = coplanars[i]
            if nonParallel(vector, mesh_primitives[0]): 
                mesh_primitives.append(vector)
                break
        mesh_primitives = np.round(mesh_primitives, 10)
        
        return mesh_primitives
    
    
    def _getMeshReciprocalVectors(self):
        """
        Get Reciprocal Lattice vectors for the lattice planes parallel 
        to the surface

        Returns
        -------
        b1 : array
             Reciprocal lattice vector
             
        b2 : array
             Reciprocal lattice vector
        """
        (a1, a2) = self.meshPrimitives
        a3 = self.surfaceNormal
        vol = abs(a1 @ np.cross(a2, a3))
        
        b1 = (2*np.pi/vol)*np.cross(a2, a3)
        b2 = (2*np.pi/vol)*np.cross(a3, a1)
        
        return (b1, b2)
    
    
    
    def _buildSlabCell(self):
        """
        Builds the full slab unit cell.

        Returns
        -------
        slabCell : list
                list of all atoms in the full slab unit cell 
        fullSlabVectors : tuple
                        3-tuple containing the large slab lattice vectors
        """
        slabCell = []
        for lz in range(self.numCells):
            for atom in self.bulk._unitCell:
                position = atom.coords_cartesian + lz*self.zPrimitive
                fractional_coords = self._cartToFrac(position)
                element = atom.element
                # if atom is at top of slab cell, put it at bottom instead
                # this prevents a non-Hermitian error
                if abs(np.round(fractional_coords[2], 9))==1.0:
                    fractional_coords[2] = 0.0
                    slabCell.insert(0, Atom(element, fractional_coords))
                else:
                    slabCell.append(Atom(element, fractional_coords))
        
        (s1, s2, s3) = self.adaptedLatticeVectors
        s3 = s3*self.numCells
        
        fullSlabVectors = (s1, s2, s3)
        
        return slabCell, fullSlabVectors
                
                
                
    
    def _cartToFrac(self, 
                    vector):
        """
        Converts a vector in Cartesian coordiantes to fractional coordinates
        for the full slab unit cell.
        
        NOTES:
        This code works using the vectors formed by cross products:
                sigma_i = a_j x a_k  (even permutations of i,j,k)
        where the a_i are the lattice vectors forming the unit cell 
        parallelepiped. These satisfy
                a_i @ sigma_j = (cellVol)*delta_ij
        where cellVol is the unit cell volume and delta_ij is the Kronecker
        delta. Thus if R = c_1 a_1 + c_2 a_2 + c_3 a_3, the fractional 
        coefficients ci are found as 
                c_i = R @ sigma_i / (cellVol)

        Parameters
        ----------
        vector : array
                3-Vector in cartesian coordinates

        Returns
        -------
        fractional_coords : array
                            vector represented in fractional coordinates
                            according to the full slab unit cell
        """
        (s1, s2, s3) = self.adaptedLatticeVectors
        s3 = s3*self.numCells
        slabCellVectors = [s1, s2, s3]
        cellVolume = abs(la.det(slabCellVectors))
        
        sigma1 = (1/cellVolume) * np.cross(s2, s3)
        sigma2 = (1/cellVolume) * np.cross(s3, s1)
        sigma3 = (1/cellVolume) * np.cross(s1, s2)
        sigma = [sigma1, sigma2, sigma3]
        
        fractional_coords = [vector @ sigma[i] for i in range(3)]
        fractional_coords = np.array(fractional_coords)
        
        return fractional_coords
    
    
    def getSlabCouplingArray(self, 
                              couplingArray):
        """
        Returns the coupling array for rigid ion phonon calculations. 
        
        NOTE: The couplings for the surface are at 'the surface' of the matrix
              i.e. they are on the edge rows and columns

        Parameters
        ----------
        couplingArray : array_like
                        Array containing the couplings between all atoms in
                        the bulk unit cell.
                        element [i,j] should have the couplings between
                        atom_i and atom_j as listed in the unitCell

        Returns
        -------
        slabCouplings : list
                        Nested lists containing the couplings between
                        atoms in the slab cell
        """

        n = self.bulk.atomsPerUnitCell
        slabCouplings = []
        
        for i in range(self.atomsPerSlabCell):
            slabCouplings.append([])
            for j in range(self.atomsPerSlabCell):
                slabCouplings[i].append( couplingArray[i%n][j%n] )
                
        return slabCouplings
    
     
    
            
        
# %%
        
#from latticeDynamics.lattice import Lattice

# =============================================================================
# a = 6.6
# latvecs = [(1/2, 1/2, 0) , (1/2, 0, 1/2) , (0, 1/2, 1/2)]
# unitCell = [('Cd', np.array((0, 0, 0))),
#           ('Te', np.array((1/4, 1/4, 1/4)))]
# latvecs = [a*np.array(v) for v in latvecs]
# lattice = Lattice(unitCell, latvecs)
# 
# slab = Slab(lattice, '211', 4)    
# 
# planes = slab._getLatticePlanes()    
# print('Surface Normal:\n',slab.surfaceNormal)
# print('\nMesh Primitive Vectors:\n', slab.meshPrimitives)
# print('\nOut of Plane Primitive Vector:\n', slab.zPrimitive)
# print('\nDistinct Planes:')
# for v in planes:
#     print(v)
# print('\nSlab:')
# print(slab)
# slab.getNeighbors(4)
# =============================================================================


