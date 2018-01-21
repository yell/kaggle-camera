#!/bin/bash
n_restarts='12';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run_pretrained.py --batch-size 20 --loss hinge --epochs 150 --resume-from ../models/d4/
    sleep 5;
done
