#!/bin/bash

for i in $(seq 0 5)
do
  for method in naive lip
  do
    exp="${i}-${method}"
    "./bin/jo_q33/${exp}" > "${exp}.txt"
  done
done
