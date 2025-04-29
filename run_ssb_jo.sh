#!/bin/bash

for q in 21 22 23 31 32 33 34; do
  for i in $(seq 0 5)
  do
    for method in naive lip
    do
      exp="${i}-${method}"
      ./bin/ssb-jo/q${q}/${exp} --t=5 > "q${q}-${exp}.txt"
    done
  done
done
