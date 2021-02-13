# A library to construct neural network for observables
For now we stick to the basic fully-connected feed-forward neural network

Each element of an observable tensor usually carries a certain symmetry. In order to describe that symmetry properly, we define such an elementary network:
* The input layer neurons share a same irreducible
* Only the totally symmetric irreducible has bias
* The activation functions have to be odd, except for the totally symmetric irreducible (for now we stick to `tanh`)
* The output is a scalor

An observable tensor is essentially a collection of its elements, each is the output of a network

This library assumes the 1st irreducible to be the totally symmetric one