#!/bin/bash

if [[ -n simpleNoise.csv ]];
then
	rm simpleNoise.csv
fi

cd ../
echo "fileName,label,fileStarter" > simpleNoise.csv
read -p "Enter NoiseFiles number : " N
shuf -n $N noise_meta.csv >> simpleNoise.csv
