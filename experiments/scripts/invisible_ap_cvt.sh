#!/bin/bash
#$1 Fold $2 epoch $3 experiment

baseDir='.'

python $baseDir/tools/test_actpred.py \
  --net output/invisible-ap-$3/invisible_train_$1/model_epoch$2.pth \
  --imdb invisible_test_$1 \
  --seqdb invisible_user$1_test \
  --cfg $baseDir/experiments/cfgs/invisible-ap-$3.yml

