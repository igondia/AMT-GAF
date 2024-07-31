#!/bin/bash

#$1 Fold $2 config
baseDir='.'

python $baseDir/tools/train_actpred.py \
  --imdb gitw_train_$1 \
  --seqdb gitw_train_$1 \
  --cfg $baseDir/experiments/cfgs/gitw-ap-$2.yml \
  --epochs 101
 
  
