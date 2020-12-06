#!/usr/bin/env bash


while [ 1 ]
do
  echo "begin new download process"
  aria2c -x5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
  if [[ $? -eq 0 ]]; then
    exit
  fi
  sleep 1800;
done
