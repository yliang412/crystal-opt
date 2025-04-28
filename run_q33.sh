#!/bin/bash

for i in $(seq 0 5)
do
  for method in naive lip
  do
    exp="${i}-${method}"
    ./bin/ssb-jo/q33/${exp} --t=5 > "${exp}.txt"
  done
done
