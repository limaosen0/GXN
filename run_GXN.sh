#!/bin/bash

# input arguments
DATA="${1-DD}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-0}  # which fold as testing data
GPU=${3-3}
test_number=${4-0}  # if specified, use the last test_number graphs as test data

# general settings
gpu_or_cpu=gpu
CONV_SIZE="32-32-32-1"
k1=0.8
k2=0.7
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
bsize=20  # batch size
dropout=True
cross_weight=1.0
fuse_weight=0.9
Rhop=1
data_root='/DATA2/data/msli/GraphClassificationData_'

# dataset-specific settings
case ${DATA} in
IMDBBINARY)
  num_epochs=200
  learning_rate=0.001
  sortpooling_k=0.9
  k1=0.8
  k2=0.5
  ;;
IMDBMULTI)
  num_epochs=200
  learning_rate=0.001
  sortpooling_k=0.9
  k1=0.8
  k2=0.7
  ;;
COLLAB)
  num_epochs=100
  learning_rate=0.001
  sortpooling_k=0.9
  k1=0.9
  k2=0.5
  ;;
DD)
  num_epochs=100
  learning_rate=0.0005
  k1=0.8
  k2=0.6
  ;;
PROTEINS)
  num_epochs=100
  learning_rate=0.001
  sortpooling_k=0.6
  k1=0.8
  k2=0.7
  ;;
ENZYMES)
  num_epochs=500
  learning_rate=0.0001
  sortpooling_k=0.8
  k1=0.7
  k2=0.5
  ;;
*)
  num_epochs=500
  learning_rate=0.00001
  ;;
esac


CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
      -seed 1 \
      -data $DATA \
      -fold $fold \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -sortpooling_k $sortpooling_k \
      -k1 $k1 \
      -k2 $k2 \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $gpu_or_cpu \
      -dropout $dropout \
      -test_number ${test_number}\
      -cross_weight $cross_weight\
      -fuse_weight $fuse_weight\
      -Rhop $Rhop \
      -data_root $data_root