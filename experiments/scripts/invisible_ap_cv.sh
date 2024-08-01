#!/bin/bash


#$1 Fold $2 experiment

baseDir='.'

python $baseDir/tools/train_actpred.py \
  --imdb invisible_train_$1 \
  --seqdb invisible_user$1_train \
  --cfg $baseDir/experiments/cfgs/invisible-ap-$2.yml \
  --epochs 41
  
