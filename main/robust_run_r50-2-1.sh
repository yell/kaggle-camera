#!/bin/bash
sleep 9000;
python run_pretrained.py --fold 1 --model resnet50 --batch-size 24 --epochs 150 --lr 5e-5 2e-3 --model-dirpath r50-2-1/
n_restarts='32';
[[ -z $1 ]] || n_restarts=$1
for i in `seq 1 ${n_restarts}`; 
do
    python run_pretrained.py --model resnet50 --batch-size 24 --epochs 150 --resume-from r50-2-1/
    sleep 5;
done
