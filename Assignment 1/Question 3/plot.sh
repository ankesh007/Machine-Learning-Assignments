#!/bin/bash

python3 LogisticRegression.py ../ass1_data/logisticX.csv ../ass1_data/logisticY.csv 0.1
echo "Done"
python3 LogisticRegression.py ../ass1_data/logisticX.csv ../ass1_data/logisticY.csv 0.001
echo "Done"
python3 LogisticRegression.py ../ass1_data/logisticX.csv ../ass1_data/logisticY.csv
echo "Done"
python3 LogisticRegression.py ../ass1_data/logisticX.csv ../ass1_data/logisticY.csv 0.000000001
echo "Done"

