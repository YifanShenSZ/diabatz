# Symmetry adaptation and scale (SAS) for internal coordinate
An internal coordinate system has no norm since some are lengthes and others are angles. To define a norm we nondimensionalize lengthes by dividing reference lengthes

The dependency of molecular properties on some internal coordinates vanishes in the dissociation limit, so the corresponding internal coordinates should be scaled

Molecular properties usually carry some symmetry arising from the identity of the nucleus, which is called the complete nuclear permutation inversion (CNPI) group. To describe the CNPI symmetry correctly, the internal coordinate system must be adapted, letting each coordinate carries a certain irreducible

## Usage
`SASICSet` is the engine class. An instance can be constructed by `SASICSet(format, IC_file, SAS_file)`, where `format` and `IC_file` are meant to construct the parent class `IntCoordSet`, `SAS_file` is an input file defining the scale and symmetry adaptation

An example of `SAS_file` is available in `test/input/SAS.in`

## Theory
The procedure to get symmetry adapted and scaled internal coordinate (SASIC) is:
1. Get internal coordinate (IC), which is taken care of by [Torch-Chemistry](https://github.com/YifanShenSZ/Torch-Chemistry)
2. Nondimensionalize the IC to get dimensionless internalcoordinate (DIC):
* for length, DIC = (IC - origin) / origin
* for angle , DIC =  IC - origin
3. Scale the DIC to get scaled dimensionless internal coordinate(SDIC):
* if no scaler: SDIC = DIC
* elif scaler is self: SDIC = f(DIC)
* else: SDIC = DIC * g(scaler DIC); when there are multiple scalers, use their average
4. Symmetry adapted linear combinate the SDIC to get SASIC

Since the scaler is usually the bond length, we choose:
* `f(DIC) = 1 - exp(-alpha * DIC)`, as Morse potential
* `g(scaler DIC) = (1 + scaler DIC)^2 * exp(-alpha * scaler DIC)`, because the self is usually an angle associated with the scalers, and an angle should vanish in both `scaler IC -> 0` (which becomes `scaler DIC -> -1`) and `scaler -> infinity`

`g'(scaler DIC) == 0` has 2 solutions:
* scaler DIC = 0, where 0 = g(0) = g'(0)
* scaler DIC = 2 / alpha - 1, which is the maximum of g
