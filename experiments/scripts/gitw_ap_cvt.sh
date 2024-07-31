#!/bin/bash

#$1 Fold $2 epoch $3 experiment
baseDir='.'

python $baseDir/tools/test_actpred.py \
  --net output/gitw-ap-$3/gitw_train_$1/model_epoch$2.pth \
  --imdb gitw_test_$1 \
  --seqdb gitw_test_$1 \
  --cfg $baseDir/experiments/cfgs/gitw-ap-$3.yml
