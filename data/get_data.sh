#!/bin/bash
mkdir raw
cd raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00221/Reaction%20Network%20\(Undirected\).data -O Reaction%20Network%20\(Undirected\).data 
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip -O Dataset.zip
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt -O 3D_spatial_network.txt
wget https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt -O USCensus1990.data.txt
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/usps.bz2 -O usps.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2 -O usps.t.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/regression/E2006.train.bz2 -O E2006.train.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/regression/E2006.test.bz2 -O E2006.test.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/sector/sector.bz2 -O sector.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/sector/sector.t.bz2 -O sector.t.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/sector/sector.scale.bz2 -O sector.scale.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/sector/sector.t.scale.bz2 -O sector.t.scale.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/real-sim.bz2 -O real-sim.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multilabel/mediamill/train-exp1.svm.bz2 -O train-exp1.svm.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multilabel/mediamill/test-exp1.svm.bz2 -O test-exp1.svm.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2 -O mnist8m.bz2
wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale.bz2 -O mnist8m.scale.bz2
for i in ./*.bz2
do
	bzip2 -d $i
done

