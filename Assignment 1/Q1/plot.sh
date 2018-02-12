#!/bin/bash

python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True False 0.001
echo "Done"
python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True False 0.005
echo "Done"
python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True False 0.009
echo "Done"
python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True False 0.013
echo "Done"
python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True False 0.017
echo "Done"
# python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False True False 0.021
# echo "Done"
# python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True 0.025
# echo "Done"
# python3 LinearRegression.py ../ass1_data/linearX.csv ../ass1_data/linearY.csv  False False True 0.05
# echo "Done"

