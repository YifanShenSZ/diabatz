# Critical geometry search for diabatz
The search is performed by [critics](https://github.com/YifanShenSZ/critics)

Since [critics](https://github.com/YifanShenSZ/critics) adopts adiabatic representation, user has to wrap his own adiabatic quantities produced by *diabatz* into `libadiabatz`, then link *critics.exe* to it by cmake. E.g. `cmake -Dadiabatz_DIR=~/Software/Mine/diabatz/tools/critics/v0-libadiabatz/share/cmake/adiabatz ~/Software/Mine/critics`
