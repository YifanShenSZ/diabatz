# A library to construct neural networks for observables
An observable tensor is essentially a collection of its elements, each of which is the output of a network

Each element of an observable tensor usually carries a certain symmetry. In order to describe that symmetry properly, we define such an elementary network:
* The input layer neurons share a same irreducible
* Only the totally symmetric irreducible has bias
* The activation functions have to be odd, except for the totally symmetric irreducible
* The output is a scalor

## Assumption
For now we stick to the basic fully-connected feed-forward neural network

For now we stick to `tanh` for activation function

The 1st irreducible is assumed to be the totally symmetric one