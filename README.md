# Time series models for sediment yield

This repo includes my modeling as a part of an AI working group of postdocs at University of Michigan.

Procedure
---
1. `clean.ipynb` : some initial preprocessing
2. `transform.ipynb` : make log transforms of the target, make date variables
3. `standard-scaling.ipynb` : object to standardize for NN learning
4. `tensorize-slices.ipynb` : format the time series for RNNs
5. `fit-lstm.ipynb` : build your models
