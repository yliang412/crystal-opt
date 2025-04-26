#!/bin/bash

# Start from dim table size of 128K
dim=131072
for i in $(seq 1 12)
do
  echo "Running test with dim table size = ${dim}"
  ./bin/ops/join --d=${dim} >> "sj_${dim}.txt"
  # Start from bloom filter size of 128K
  bf=524288
  for j in $(seq 1 10)
  do
    echo "\nbf = ${bf}" >> "sj_${dim}.txt"
    ./bin/ops/bloom_join --d=${dim} --b=${bf} >> "sj_${dim}.txt"
    bf=$((bf * 2))
  done
  dim=$((dim * 2))
done
