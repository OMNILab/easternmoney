#!/bin/bash

for ((i = 1 ; i < 20 ; i = i + 1));do

product=`gawk -v x=$i -v y=0.05 'BEGIN{printf "%.2f\n",x*y}'`
echo $product
python easternmoney_split.py $product
done
