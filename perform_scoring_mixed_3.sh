#! /bin/bash

########################################
# MIXED-3 Hyperparameter Search MSE Exps
########################################

for model in \
MP329 \
MP330 \
MP331 \
MP332 \
MP333 \
MP334 \
MP335 \
MP336 \
MP337 \
MP338 \
MP339 \
MP340
do
    .env/bin/python dogo/score_model_orig.py $model
done