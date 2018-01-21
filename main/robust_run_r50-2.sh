#!/bin/bash
n_restarts='32';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run_pretrained.py --model resnet50 --batch-size 24 --epochs 150 --resume-from ../models/r50-2/
    sleep 5;
done
