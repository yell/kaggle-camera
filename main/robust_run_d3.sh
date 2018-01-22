#!/bin/bash
sleep 9500;
n_restarts='32';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run_pretrained.py --batch-size 20 --epochs 150 --resume-from d3/
    sleep 5;
done
