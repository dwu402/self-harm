#!/bin/bash
set -e
DATADIR="auto-got-data-$(date +%Y%m%dZ%H%M%S)"
mkdir $DATADIR
cd $DATADIR

if [ "$#" -ne 1 ]; then
  echo "Defaulting to sources.txt"
  SOURCES="sources.txt"
else
  SOURCES=$1
fi

while read SOURCE; do
  curl -OLJ $SOURCE
done < ../$SOURCES
cd - | echo
echo "Sources gathered in directory"
echo $DATADIR
