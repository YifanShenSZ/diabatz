# Theory
From quantum chemical calculations, we can obtain a few lowest adiabatic states, i.e. eigenvectors of electronic Hamiltonian operator. The goal of diabatization is to get a diabatic representation that can reproduce the ab initio quantities.

Naturally, there appears to be 2 ways to perform diabatization:
1. Start from the ab initio adiabatic states, rotate them to diabatic
2. Start from a model diabatic representation, fit it to the ab initio quantities

Way 1 has been proven to be impossible:
* Consider `N` electronic states
* The sufficient and necessary condition of a diabatic representation is `the gradient of basis vectors = 0`, which corresponds to `coordinate dimension * N (N - 1) / 2` equations
* To rotate `N` states, we need to define an orthogonal matrix, which has `N (N - 1) / 2` independent elements
* As a result, only when `coordinate dimension = 1` can the sufficient and necessary condition be satisfied

Way 2 rigorously satisfies te diabatic condition, although it may not be able to reproduce ab initio adiabatic quantities as precise as way 1. This is a tolerable trade off, since we can always reduce the fitting error with more flexible fitting expansion

Concretely, way 2 fits the adiabatic energy, energy gradient and nonadiabatic coupling. Energy must be fitted, of course. Nonadiabatic coupling has to be included as well, otherwise we can end up with N uncoupled potential energy surfaces. Energy gradient is of same complexity as nonadiabatic coupling, so it is included as a why not.

Standard machine learning works only on how to fit target function (energy); since we are also fitting the target derivative (energy gradient and nonadiabatic coupling), we will need slightly difference experience

## Machine learning target derivative
For now we consider a neural network with only 1 hidden layer, no bias. Let `Ain` be the input layer weights (which is a matrix), `Aout` be the output layer weights (which is a vector), `x` be the features (which is a vector), `y` be the target, `a` be the activation function, a standard machine learning target is
```
z = Ain . x        (1)
y = Aout . a(z)
```
Now we also want the target derivative over `r`
```
▽_r y = (▽_r x)^T . Ain^T . ▽_z a(z) . Aout    (2)
```
To some extent, we can consider a derivative network, where `▽_r y` is the target, `▽_r x` is the feature, `▽_z a(z)` is the activation:
* feature scaling needs to consider `▽_r x`
* given `x`, `Ain^T . ▽_z a(z) . Aout` is the key to fitting `▽_r y`

Feature scaling has the form of `X = (x - shift) / width`. `shift` does not affect `▽_r X`, only `width`:
```
▽_r X = (▽_r x) / width    (3)
```
To normalize `▽_r X`, we need metric to define `width`. `S = (▽_r x) . (▽_r x)^T)` is an appropriate candidate as `S[i][i] = ||▽_r x[i]||^2`, so we let `width[i]` be the square root of the maximum of `S[i][i]` among the training set

Assume `▽_z a(z) = 1` (which is true for tanh when `z = 0`), equation (2) tells us that `Ain^T . Aout = ▽_r y / (▽_r x)^T`:
* when there is only 1 hidden neuron, `Ain^T[i] * Aout = (▽_r y / (▽_r x)^T)[i]`, which means that even with regularization `Ain^T[i]` and `Aout` cannot be very small simultaneously
* so the only way to produce small weight elements is to increase the number of hidden neurons: the more terms involved in the `.` of `Ain^T . Aout`, the smaller each term can possibly be