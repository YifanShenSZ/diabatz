# Version 0.1.2
Version 0.1.0 + replace trust region with pytorch optimizers

This version is meant to save memory and to walk out of deep narrow valley (usually overfitting local minimum)

Using different weights for different data points deteriorates random optimizers: gradient would fluctuate more across mini-batchs.

A workaround is to use only interger weight then treat it as duplicate, i.e. an interger weight is equivalent to how many duplicates of a data point appear in the training set

So this version would ceil all weights