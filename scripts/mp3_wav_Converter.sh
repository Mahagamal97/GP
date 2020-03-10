#!/bin/bash

if [[ -d ../wav_dir ]];
then
	rm -r ../wav_dir
	mkdir ../wav_dir
else
	mkdir ../wav_dir
fi 

cd ../mp3_dir
for fileName in *.mp3;
do
	name=$(echo "$fileName" | cut -f 1 -d '.')
	#echo "$name"
	ffmpeg -i "$fileName" "../wav_dir/$name.wav";
done