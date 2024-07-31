#!/bin/bash

#$1 Fold $2 experiment

baseDir='.'

python $baseDir/tools/train_actpred.py \
  --imdb sharon_train_$1 \
  --seqdb sharon$1_train \
  --cfg $baseDir/experiments/cfgs/sharon-ap-$2.yml \
  --epochs 101
   
  
