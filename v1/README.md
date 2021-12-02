# Version 1
Version 0 + pytorch optimizers

Using different weights for different data points deteriorates random optimizers. Gradient would fluctuate more across mini-batchs.

A workaround is to use only interger weight then treat it as duplicate, i.e. an interger weight is equivalent to how many duplicates of a data point appear in the training set

So this version would ceil all weights