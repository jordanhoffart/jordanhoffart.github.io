#!/usr/bin/bash
root=$(pwd)
for i in $(find $root -type d)
do
	cd $i
	latexmk
	cd $root
done
