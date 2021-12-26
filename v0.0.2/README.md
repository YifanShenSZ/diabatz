# Version 0.0.2
Version 0.0.0 + replace trust region with pytorch optimizers

This version is meant to save memory and to walk out of deep narrow valley (usually overfitting local minimum)

## Issue: weight
Using different weights for different data points deteriorates random optimizers: gradient would fluctuate more across mini batchs.

A workaround is to use only interger weight then treat it as duplicate, i.e. an interger weight is equivalent to how many duplicates of a data point appear in the training set

So this version would ceil all weights

## Issue: data type
Also, the type of different data points creates gradient fluctutation

The best way is to let each mini batch contain all types of data points, so each mini batch is still a good estimation of the total batch

However, I do not know how to implement a data loader for multiple types of data points

So, for now we should only use one type of data at a time
