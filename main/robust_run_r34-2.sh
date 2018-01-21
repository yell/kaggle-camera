#!/bin/bash
n_restarts='12';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run_pretrained.py --model resnet34 --batch-size 64 --epochs 150 --resume-from ../models/r34-2/
    sleep 5;
done
