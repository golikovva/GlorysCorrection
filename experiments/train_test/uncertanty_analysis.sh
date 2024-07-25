#!/bin/bash
if [ -n "$1" ]
then
numruns=$1
else
numruns=5
fi
echo Run main for $numruns times
for (( i=1; i <= numruns; i++ ))
do
python main.py
done