#!/bin/bash
n_restarts='12';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run.py --batch-size 24 --resume-from c5/
    sleep 5;
done
