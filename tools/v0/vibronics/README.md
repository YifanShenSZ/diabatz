# Generate *vibronics* input

## Phonon suggestion
A vibronic spectrum usually has the brightest line near the vertical excitation, which emphasizes the overlap between the final-state-biased vibrational basis and the initial vibrational state

Our goal is to find the minimum basis producing a satisfying overlap (empirically > 90%), and this is an integer nonlinear optimization subject to inequality constraint

A rigorous evaluation solution will be too expensive, so we make 2 approximations:
1. In most cases the precursor is at vibrational ground state, which is a gaussian with covariance matrix diagonal in precursor normal coordinate. We approximate the overlap constraint by 'the basis covers some sigma eclipse', by analogy to '2 sigma covers 95%' in single variate case
2. Define the coverage of a 1 dimensional basis function as its standard deviation, then the total coverage is a cuboid with each edge equals to the widest basis along this direction

Certainly, the smallest cuboid must be tangential to the eclipse, so we only have to solve the lower and upper bound of the eclipse along each residual normal coordinate

Now the problem can be solved by a common real nonlinear optimization subject to equality constraint: Minimize and maximize the component along each residual normal coordinate, subject to staying on the 2 sigma eclipse

## Reference
> 1. [Y. Shen and D. R. Yarkony, J. Phys. Chem. Lett. 2020, 11, 17, 7245–7252](https://doi.org/10.1021/acs.jpclett.0c02199)