#!/bin/bash

if [[ -n clean.csv ]];
then
	rm clean.csv
fi

cd ../clean_train

ls | while read files;
do
	if [ ${files: -4} == ".wav" ];
	then
		echo "$files" >> ../clean.csv
	fi
done