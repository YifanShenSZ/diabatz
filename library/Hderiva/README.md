# A library for Hamiltonian derivatives
All the user need to code is a (backwardable) `Hd`, since this library will take care of every other necessary quantities, unless the user want some specialization

Nomenclature:
* `x` stands for the coordinate vetor
* `c` stands for the trainable parameters vector
* Anything without a suffix is an operator. E.g. `H` is Hamiltonian operator, `d / dx * H` is the gradient of Hamiltonian operator over geometry (so it's still an operator)
* Anything with a suffix is a matrix, with the suffix indicating the representation. E.g. `Hd` is the diabatic Hamiltonian matrix, `d / dx Hd` is the gradient of the diabatic Hamiltonian matrix over geometry (so it's a matrix with vectors as its elements), `(d / dx * H)a` is the adiabatic `d / dx * H` matrix (with vectors as its elements)

Note that the matrix of the gradient operator is different from differentiating the matrix, since the latter additionally differentiates the basis vectors of the matrix representation. E.g. `(d / dx * A)a != d / dx Aa`:
* `(d / dx * A)a = U^T . (d / dx * A) . U`, where `H . U = U . E`
* `d / dx Aa = U^T . (d / dx * A) . U + [Aa, M]`, where `d / dx * U = U . M`
The only exception is the diabatic representation, since the diabatic basis is constant. That's why we love [*diabatz*](https://github.com/YifanShenSZ/diabatz)! An operator is truely equivalent to its matrix form in diabatic representation, even under differentiating!

Common representations:
* Diabatic  representation (suffix `d`)
* Adiabatic representation (suffix `a`)
* Composite representation (suffix `c`)

Available quantities:
1. `d / dx * Hd`, `d / dc * Hd`, `d / dc * d / dx * Hd`
2. `d / dc * (d / dx * H)a`
3. `d / dc * Hc` and `d / dc * (d / dx * H)c`

## Specialization
In specific applications, some parts of the `x -> Hd` graph has more effcient ways to construct Jacobians than backward propagation, so user may provide these Jacobians for better performance

If `Hd` is computed from [*obnet*](https://github.com/YifanShenSZ/diabatz/tree/master/library/obnet), then user may provide:
* The Jacobian of the input layer over `x`

If `Hd` is computed from [*DimRed*](https://github.com/YifanShenSZ/diabatz/tree/master/library/DimRed) and [*obnet*](https://github.com/YifanShenSZ/diabatz/tree/master/library/obnet), then user may provide:
* The Jacobian of the input layer over the reduced coordinate `r`
* The Jacobian of `r` over `x`
* The 2nd-order Jacobian of `r` over `x` and *DimRed* parameters