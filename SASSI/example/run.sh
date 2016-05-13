#!/bin/bash
APP="kernelMatrixMul"
#make clean
#make mm
cuobjdump ./$APP -xelf all
mv ?.sm_35.cubin sm_35.cubin
mv ?.sm_52.cubin sm_52.cubin
nvdisasm sm_35.cubin -cfg > sm_35.dot
nvdisasm sm_52.cubin -cfg > sm_52.dot

echo $1

	if [ $1 == 35 ];
	then
		xdot sm_35.dot &
	else
		if [ $1 == 52 ];
		then
			xdot sm_52.dot &
		fi
	fi
rm -rf *.cubin
