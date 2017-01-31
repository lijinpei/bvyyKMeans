#!/bin/bash
for k in 200 400 600 800 1000
do
date && ../src/a.out -n 6412 -d 55198 -f raw/sector.scale  -k $k -i -1  -p 1e-4  --sparse --kpp -s noc_sector_$k.$bs.out -y >not_sector_yinyang_$k.output 2>&1 && date &
done
