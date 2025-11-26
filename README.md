# Time series models for sediment yield

This repo includes my modeling as a part of an AI working group of postdocs at University of Michigan.

Procedure
---
1. `clean.ipynb` : some initial preprocessing
2. `transform.ipynb` : make log transforms of the target, make date variables
3. `standard-scaling.ipynb` : object to standardize for NN learning
4. `tensorize-slices.ipynb` : format the time series for RNNs
5. `fit-kfold.ipynb` : build your models
6. Or, systematically explore architectures and hyperparameters.
    - Modify `modeling.sh` which creates JSON files in `configs/`
    - Modify `kfolding.sh` to match `modeling.sh` and `slurm` details
    - `bash kfolding.sh` which creates training details in `outputs/`
7. Summarize results into `parsed_logs/` with `python results.py`

For each fold, it could take multiple hours on a GPU node!
