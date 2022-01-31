# SASDIC
An internal coordinate system has ill-defined unit, since some are lengthes and others are angles. We nondimensionalize lengthes by dividing reference lengthes, so the unit is always 1

The dependency of molecular properties on some internal coordinates vanishes in the dissociation limit, so the corresponding internal coordinates should be scaled

Molecular properties usually carry some symmetry arising from the identity of the nucleus, which is called the complete nuclear permutation inversion (CNPI) group. To describe the CNPI symmetry correctly, the internal coordinate system must be adapted, letting each coordinate carries a certain irreducible

In conclusion, we need symmetry adapted and scaled dimensionless internal coordinate (SASDIC)

## Usage
`SASDICSet` is the engine class. An instance can be constructed by `SASDICSet(format, IC_file, SAS_file)`, where `format` and `IC_file` are meant to construct the parent class `IntCoordSet`, `SAS_file` is an input file defining the scale and the symmetry adaptation

An example of `SAS_file` is available in `test/input/SAS.in`

## Theory
The procedure to get SASDIC is:
1. Get internal coordinate (IC), which is taken care of by [Torch-Chemistry](https://github.com/YifanShenSZ/Torch-Chemistry)
2. Nondimensionalize the IC to get dimensionless internalcoordinate (DIC):
* for length, DIC = (IC - origin) / origin, so DIC ∈ [-1, infinity)
* for angle , DIC =  IC - origin
3. Scale the DIC to get scaled dimensionless internal coordinate (SDIC)
4. Linearly combine the SDIC in a symmetry-adapted way to get SASDIC

## Implementation
Let scaling function be f:
* If self == other: SDICs[self] = f(DICs[other])
* else:             SDICs[self] = f(DICs[other]) * DICs[self]

Available scaling functions:
* `1-exp(-a*x)`, which produces Morse potential
* `exp(-a*x)*(1+x)^b`, which resembles the radial distribution of hydrogen orbitals, reaches maximum at `x = b/a - 1` and approaches 0 at `-1 <- x` and `x -> infinity`
