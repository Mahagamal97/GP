#!/bin/bash

if [[ -n dataScripting.csv ]];
then
	rm dataScripting.csv
fi

cd ../noiseTrain

ls | while read FOLDERS ;
do
	#echo "$PWD"
	#pwd
	cd $FOLDERS
	ls | while read FILES ; 
	do
		echo "$FILES,$FOLDERS" >> ../../dataScripting.csv
	done
	cd ../
done 
