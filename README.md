# lattice-dynamics
This repo is for developing a package to aid in computations in phenomenological models of lattice dynamics. The goal is to be able to import this in a jupyter notebook and then be able to build and tweak a model easily and with clean looking code.
Right now there are a few classes:

## The Core Classes

- Lattice

This class is for handling everything to do with atomic positions in the crystal. Conveniently, there already exists a python package called [Crystals](https://pypi.org/project/crystals/) which allows for getting the whole crystal symmetry and atomic positions by giving it 1) the fractional atom positions in the unit cell and 2) the primitive lattice vectors. So Lattice inherits from the Crystal class in the Crystals package. On top of everything it inherits, a Lattice instance also can get information about near neighbors, visualize neighbors, and can give the mass matrix used in phonon dispersion calculations.

- RigidIon

This class contains all the functions needed for calculating the rigid ion contribution to the dynamical matrix. It takes as one of
its parameters a Lattice instance so it can determine near neighbors.

- Coulomb

This class contains all the functions needed for calcalating the long-range Coulomb contribution to the dynamical matrix. 
The functions are an implementation of the Ewald method.

- Model

This is the 'interface class' alongside Lattice. When actually building a model, this is the class which puts together
all the pieces. It is meant to handle the RigidIon and Coulomb classes under the hood to simplify things. It constructs the full
dynamical matrix for the model and can diagonalize it to give the phonon dispersion and normal modes. It also is meant for any 
plotting functions. 

Since the Coulomb calculation can be relatively slow, there are options for storing Coulomb contributions previously calculated
along a set path through reciprocal space.


