# lattice-dynamics
This repo is for developing a python package for phenomenological models of lattice dynamics. In particular the focus is on using both
bulk and slab calculations to determine surface phonon modes. The goal is to be able to import this in a jupyter notebook and then be able to build 
and tweak a model easily and with clean looking code.
Right now there are a few classes:

## The Core Classes

- `Lattice`

This class is for handling everything to do with atomic positions in the bulk crystal. Conveniently, there already exists a python 
package called [Crystals](https://pypi.org/project/crystals/) which allows for getting the whole crystal symmetry and atomic positions 
by giving it 1) the fractional atom positions in the unit cell and 2) the primitive lattice vectors. So `Lattice` inherits from the `Crystal` 
class in the Crystals package. On top of everything it inherits, a `Lattice` instance also can get information about near neighbors, 
visualize neighbors, and can give the mass matrix used in phonon dispersion calculations.

- `Slab`

This class is the analog of `Lattice` but for slab geometries (a crystal which is infinite in two dimensions but finite in the third).
Given a surface defined by the Miller indices (*h, k, l*), `Slab` will construct a "surface-adapted" set of primitive lattice vectors
for the bulk crystal defined so that two primitive vectors are parallel to the plane of the surface and one is out of the plane. With
these, it builds a `Crystal` object which has as its unit cell the full thickness of the slab. This bulk lattice is then cut into a slab
by restricting interactions to only be between atoms within the desired volume.

**NOTE**: Whereas surfaces in cubic crystal are named with respect to the conventional unit cell instead of the primitive cell, `Slab` objects
only work with miller indices which are coefficients of *primitive* reciprocal lattice vectors.


- `RigidIon`

This class contains all the functions needed for calculating the short-range force contribution to the dynamical matrix. It takes as one of
its parameters a `Lattice` instance so it can determine near neighbors.

- `Coulomb`

This class contains all the functions needed for calcalating any long-range Coulomb contributions to the dynamical matrix. 
The functions are an implementation of the Ewald method.

- `Model`

This is the 'interface class' alongside `Lattice` and `Slab`. When actually building a model, this is the class which puts together
all the pieces. It is meant to handle the `RigidIon` and `Coulomb` classes under the hood to simplify things. It constructs the full
dynamical matrix for the model and can diagonalize it to give the phonon dispersion and normal modes. It also is meant for any 
plotting functions. 

Since the Coulomb calculation can be relatively slow, there are options for storing Coulomb contributions previously calculated
along a set path through reciprocal space.


