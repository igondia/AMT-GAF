#!/bin/bash

#$1 Fold $2 epoch $3 experiment

baseDir='.'


python $baseDir/tools/test_actpred.py \
  --net output/sharon-ap-$3/sharon_train_$1/model_epoch$2.pth \
  --imdb sharon_test_$1 \
  --seqdb sharon$1_test \
  --cfg $baseDir/experiments/cfgs/sharon-ap-$3.yml

