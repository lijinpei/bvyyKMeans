#!/bin/bash
bs=200
for k in 200 400 600 800 1000
do
date
../src/a.out -n 6412 -d 55198 -f raw/sector.scale  -k $k -i -1  -p 1e-4  --sparse --kpp -q noc_sector_$k.$bs.out --block_size $bs >not_sector_optimized_$k.$bs.output 2>&1
date
done



