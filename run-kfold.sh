#!/bin/bash
idx=$1
papermill -p params_file configs/$idx.json -p output_prefix outputs/fit-kfold-$idx fit-kfold.ipynb ran/fit-lstm-kfold-$idx.ipynb
