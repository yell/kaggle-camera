#!/bin/bash
n_restarts='12';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run_pretrained.py --model resnet50 --batch-size 24 --epochs 150 --resume-from r50-2/
    sleep 5;
done