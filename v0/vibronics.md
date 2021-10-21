# Use *diabatz* with *vibronics*
*vibronics* requires polynomial-expanded diabatic Hamiltonian, so the network must have no hidden layer

*vibronics* uses point group at the final-state geometry, so the CNPI group must be isomorphic to the point group there

*vibronics* uses normal coordinate, so the polynomial set will be translated and rotated from the internal coordinate to the normal coordinate, which means the polynomial definition must meet *Torch-Chemistry* advanced requirement