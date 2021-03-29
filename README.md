# Diabatz
A software for nonadiabatic chemistry in diabatic representation

The core of diabatic representation is its constant basis vectors. The most amazing properties derives from this core is perhaps the equivalence between an operator and its matrix form in diabatic representation. Consider an operator A, whose diabatic matrix form is Ad:
* Applying Ad to wave function vector is isomorphic to applying A to wave function, of course
* ▽Ad equals to the diabatic ▽A matrix. In general representations, however, you need another term accounting for the gradient of basis vectors

Usually a representation is defined to diagonalize some operator, e.g. adiabatic representation diagonalizes Hamiltonian operator. That can become problematic, however, when the operator has (near) degenerate eigen values, since the corresponding eigen vectors can arbitrarily mix, introducing <span style="color:red">singularity to the gradient of basis vectors</span>