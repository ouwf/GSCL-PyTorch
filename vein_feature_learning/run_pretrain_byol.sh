#!/bin/bash

dataset=FVUSM
network=resnet18
loss=byol
max_epoch=80
batch_size=64
trainset=$(realpath "${1}")
testset=$(realpath "${2}")
num=20000
seed=99
lr=0.03
wd=0.0004
timestamp=$(date +%s)
python3 -u ./main_ssl.py \
  --seed $seed \
  --dataset_name $dataset --network $network --loss ${loss} \
  --batch_size ${batch_size} --max_epoch $max_epoch \
  --trainset ${trainset} --synthetic_num ${num} \
  --testset $testset \
  --lr $lr --wd $wd \
  --save_image \
  2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}_${num}_${timestamp}.txt