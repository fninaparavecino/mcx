#!/bin/bash

user="fanny"

machine=`hostname`

	echo $machine
	../../bin/mcx -L > listcards.txt
	devcount=`grep 'Device ' listcards.txt  | wc -l`

	echo "computer: "$machine
	echo "total devices: "$devcount

	for id in $(seq $devcount)	
	do
		echo $id
		./run_benchmark1.sh -G $id > ${machine}_${id}_b1.log
	done
