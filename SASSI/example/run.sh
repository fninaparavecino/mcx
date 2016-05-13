#!/bin/bash
APP="kernelMatrixMul"

make none
./$APP
cuobjdump ./$APP -xelf all
mv ?.sm_20.cubin sm_20.cubin
mv ?.sm_35.cubin sm_35.cubin
mv ?.sm_52.cubin sm_52.cubin
nvdisasm sm_20.cubin -c -sf -cfg > sm_20.cfg
nvdisasm sm_35.cubin -c -sf -cfg > sm_35.cfg
nvdisasm sm_52.cubin -c -sf -cfg > sm_52.cfg

echo $1

if [ $1 == 20 ];
then
	xdot sm_20.cfg &
	nvdisasm -lrm count sm_20.cubin &
else
	if [ $1 == 35 ];
	then
		xdot sm_35.cfg &
		nvdisasm -lrm count sm_235.cubin &
	else
		if [ $1 == 52 ];
		then
			xdot sm_52.cfg &
			nvdisasm -lrm count sm_52.cubin &
		fi
	fi
fi
rm -rf *.cubin
