#!/bin/bash
set -e

START=$1
END=$2
INCREMENT=$3

CONFIG_TEMPLATE=$4
OUTPUT_FILE=$5

for i in $(seq $START $INCREMENT $END) ; do
  cp $CONFIG_TEMPLATE "$CONFIG_TEMPLATE"_working
  echo "rg $i" >> "$CONFIG_TEMPLATE"_working
  ./main.py -v -a lcurve -c "$CONFIG_TEMPLATE"_working -o $OUTPUT_FILE
done

./lcurve.py -f $OUTPUT_FILE
