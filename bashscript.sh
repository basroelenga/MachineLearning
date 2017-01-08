#!/usr/bin/env bash
# Runs the script that generates all the images 
# in different folders

for i in 1 2 3 4 5 6 7 8 9 
do
     ./main.py $i &
done  
