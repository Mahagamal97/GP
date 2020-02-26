#!/bin/bash

if [[ -n metaData.csv ]];
then
	rm metaData.csv
fi

cd noise_train

ls | while read FOLDERS ;
do
	#echo "$PWD"
	#pwd
	cd $FOLDERS
	ls | while read FILES ; 
	do
		echo "$FILES,$FOLDERS" >> ../../metaData.csv
	done
	cd ../
done 
