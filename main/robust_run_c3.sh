#!/bin/bash
n_restarts='12';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run.py --epochs 150 --resume-from ../models/c3/
    sleep 5;
done
