#!/bin/bash
idx=$1
papermill -p params_file configs/$idx.json -p output_file outputs/fit-kfold-$idx.txt fit-kfold.ipynb ran/fit-lstm-kfold-$idx.ipynb
