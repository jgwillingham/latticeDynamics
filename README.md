# latticeDynamics
This repo is for the development of a python package that has tools for building phenomenological models of lattice dynamics
and then calculating the phonon dispersion in the bulk and at surfaces. The goal is to be able to import this in a Jupyter notebook
and then be able to build, change, and explore a model with clean, succinct code. 

There is a focus on tools for studying surface phonons. In the future, [z2pack](http://z2pack.ethz.ch/doc/2.1/index.html) will
be incorporated into latticeDynamics allowing the study of phonon topology. Then Chern numbers of the bulk band structure and
corresponding topological surface modes can be seemlessly explored within a model.

Also to be added soon are tools for the symmetry classification of phonon modes.

## Building a Model

### 1. Crystal Structure

**Bulk Phonons**

For looking at bulk phonon structure, all information about the crystal structure is stored in an instance of the `Lattice` class. Thanks 
to a pre-existing package [Crystals](https://pypi.org/project/crystals/), most of the functionality of a `Lattice` object comes from the 
`Crystal` class. By providing 

- The fractional atomic positions in the unit cell + element symbols

- The primitive lattice vectors

the crytal's space group is determined (using [spglib](https://atztogo.github.io/spglib/)) along with symmetry operations and atomic masses.

**Surface Phonons**

For looking at surface phonons, the analog of `Lattice` is the `Slab` class. This class is intended for the study of surface phonons in the
slab geometry but is also used for semi-infinite crystals used in an iterative Green's function approach. When creating a `Slab` object, by providing

- A `Lattice` instance for the bulk structure

- A string of Miller indices indicating the surface of interest*

- The thickness of the slab (in number of unit cells)

all information about the crystal slab is determined. **NOTE**: Whereas surfaces in cubic crystal are named with respect to the conventional 
unit cell instead of the primitive cell, `Slab` objects only work with miller indices which are coefficients of *primitive* reciprocal lattice vectors.

### 2. Modeling the Lattice Dynamics (Phenomenological Parameters)

With the crystal/slab structure stored and available, we can begin modeling interaction between the atoms. There are currently two interaction classes:

- `RigidIon`: For radial short-range interactions between ions.

- `Coulomb`: For long-range Coulomb interactions in ionic crystals.

Neither of these are intended to be directly interfaced by a user. Instead, to build the model you create an instance of the `Model` class. 
By passing in an array of coupling constants for the short-range interactions, they are fully set. For the Coulomb interactions, there is an option
for setting `withCoulomb` to `True` or `False` indicating whether or not to include long-range electric effects. Then the a list of charges on the ions 
act as the phenomenological parameters.

### 3) Calculating Dispersion/Normal Modes and Plotting

With the interaction parameters specified upon instantiation of a `Model`, the dynamical matrix can be constructed and diagonalized. The `Model` class has
methods for calculating the dispersion along a path through the Brillouin zone and for plotting. 





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

- `GreensFunction`

This class contains all the methods for the iterative Green's function calculation. It uses the decimation-recursion algorithm (Sancho 1985).
All it needs to work is a slab model dynamical matrix `model.D`.


