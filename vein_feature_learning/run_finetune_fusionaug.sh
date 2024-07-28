#!/bin/bash

dataset=FVUSM
network=resnet18
loss=fusionloss
max_epoch=80
batch_size=64
trainset=$(realpath "${1}")
testset=$(realpath "${2}")
ckpt="null"
load_from=${3:-$ckpt}
seed=99
lr=0.01
timestamp=$(date +%s)
python3 -u ./main_sl.py \
  --seed $seed \
  --dataset_name $dataset --trainset ${trainset} --testset $testset \
  --network $network --loss ${loss} \
  --batch_size ${batch_size} --max_epoch $max_epoch \
  --lr $lr \
  --intra_aug --inter_aug "TB" \
  --save_image \
  --load_from $load_from \
  2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}_${timestamp}.txt