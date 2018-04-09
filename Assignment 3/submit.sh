#!/bin/bash

name="2015cs10435_ankesh_gupta"
src="source"

rm -r $name
mkdir $name

cd $name
mkdir $src
cd ../

cp *.ipynb $name/$src/
cp *.py $name/$src/
cp Assignment\ 3.pdf $name/