#!/bin/bash

cd ../

if [[ -n clean_meta.csv ]];
then
	rm clean_meta.csv
fi
echo "fileName">>clean_meta.csv 
cd clean

ls | while read files;
do
	if [ ${files: -4} == ".wav" ];
	then
		echo "$files" >> ../clean_meta.csv
	fi
done