Add a simple classifier script [options]

--threshold N[50] only use species seen in at least N data points

--target [Managment] Variable to predict ('Winery', 'Vineyard', 'Varietal', 'AlleyVine', 'Management', 'AVA', 'subAVA', 'LatitudeSite', 'LongitudeSite', 'Precip2017', 'Precip', 'Till', 'SoilFertilityMgmt', 'SoilFertMgmt', 'RS', 'Compost', 'MineralFert', 'PruneMulch)

--type [xbg] mlp,linear,xgb

--nobalance do not balance classes

--input_data File to the Species csv

--input_metadata File to the metadata text file

--output Folder to output plots and fitted model

Example

python classifier_test.py --input_data ~/vineyard/VineyardFiles/Summer/VMP-U18_fungiASVs.txt --input_metadata ~/vineyard/VineyardFiles/Summer/VMP-U18metaFixed.csv --output plots --threshold=500 --target=Till --type=xgb