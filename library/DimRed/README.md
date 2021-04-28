# A library to construct neural networks for dimensionality reduction
Symmetry adapted internal coordinate provides a more compact representation than Cartesian coordinate, but it may still contain redundancy for certain applications, where some modes are merely spectators

To obtain a most compact representation, we define a dimensionality reduction network:
* Each irreducible owns a network, whose inputs are the symmetry adapted internal coordinates of this irreducible
* Only the totally symmetric irreducible has bias
* The activation functions have to be odd, except for the totally symmetric irreducible

The dimensionality reduction network can be viewed as the encoder part of an autoencoder, whose encoder part reduces the dimension while the decoder part inverse to the original dimension, so:
* we can pretrain the dimensionality reduction network by training the corresponding autoencoder and take its encoder part
* ~~(to do) we may train the decoder part after we have obtained the dimensionality reduction network from fitting the interesting observables~~

## Assumption
For now we stick to the basic fully-connected feed-forward neural network

For now we stick to `tanh` for activation function

The 1st irreducible is assumed to be the totally symmetric one