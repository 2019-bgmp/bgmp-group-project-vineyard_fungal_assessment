#!/bin/bash

module purge 
conda activate xg26
n=1;
max=10;
while [ "$n" -le "$max" ]; do
      mkdir "test$n"
      cd "test$n"
      mkdir "plots"
      cd ..
        n=`expr "$n" + 1`;
done

input=/home/bcosgrov/bgmp/fungroup/dada2_processing/ASV-VT.txt
meta=/home/bcosgrov/bgmp/fungroup/191119_VMP-ALLmeta.csv

###NOTE: threshold is a representation of how many times a certain species/genera/etc must occur to be included in the output. 
###For more precise rarification threshold ranges should be changed to better fit the data on hand.

./classifier_test.py --input_data $input --input_metadata $meta --output test1 --target=subAVA --type=xgb --threshold=1
./classifier_test.py --input_data $input --input_metadata $meta --output test2 --target=subAVA --type=xgb --threshold=2
./classifier_test.py --input_data $input --input_metadata $meta --output test3 --target=subAVA --type=xgb --threshold=3
./classifier_test.py --input_data $input --input_metadata $meta --output test4 --target=subAVA --type=xgb --threshold=4
./classifier_test.py --input_data $input --input_metadata $meta --output test5 --target=subAVA --type=xgb --threshold=5
./classifier_test.py --input_data $input --input_metadata $meta --output test6 --target=subAVA --type=xgb --threshold=6
./classifier_test.py --input_data $input --input_metadata $meta --output test7 --target=subAVA --type=xgb --threshold=7
./classifier_test.py --input_data $input --input_metadata $meta --output test8 --target=subAVA --type=xgb --threshold=8
./classifier_test.py --input_data $input --input_metadata $meta --output test9 --target=subAVA --type=xgb --threshold=9
./classifier_test.py --input_data $input --input_metadata $meta --output test10 --target=subAVA --type=xgb  --threshold=10

