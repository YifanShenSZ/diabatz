# A library to load and process ab initio data
The most basic ab initio data is geometry, who is the variable for any observable

The geometry itself can feed unsupervised learning e.g. autoencoder

A common label is Hamiltonian (and gradient), who is classified into regular or degenerate based on degeneracy threshold. A regular Hamiltonian is in adiabatic representation, while a degenerate one is in composite representation

## Data format
Everything should be in atomic unit

A directory is a smallest unit to hold a data set. It should contain a several files:
* `geom.data`
* `energy.data`
* `cartgrad*.data`
where `*` can be `-n` or `-m-n` (`m` < `n`), for energy gradients of state `n` or nonadiabatic coupling between states `m` and `n`

Each line of `energy.data` contains the energies of each state. *abinito* infers the number of data points from its number of lines

Each line of `geom.data` contains `atom symbol, x, y, z`. *abinito* infers the number of atoms constituting the molecule from its number of lines / the number of data points

Each line of `cartgrad*.data` contains `x, y, z`, so its number of lines must equal to that of `geom.data`

## Symmetry adaptation
Until now we are using the simplest Cartesian coordinate. However, in many cases we would need symmetry adapted internal coordinate because the molecular properties
1. are invariant under translation and rotation
2. carry certain symmetry

A global description is based on CNPI group. For a specific data point, the CNPI group is mapped to point group. The mapping is defined by `CNPI2point.txt` in the data directory:
* each line defines the mapping rule of a geometry
* e.g. `1 2 -> 1 1` means CNPI group irreducible `1` and `2` become point group irreducible `1` and `1`

## Reference
1. Y. Shen and D. R. Yarkony, J. Phys. Chem. A 2020, 124, 22, 4539–4548 https://doi.org/10.1021/acs.jpca.0c02763
