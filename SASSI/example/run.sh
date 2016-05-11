#!/bin/bash
APP="kernelMatrixMul"
./$APP
cuobjdump ./$APP -xelf all
mv ?.sm_20.cubin sm_20.cubin
mv ?.sm_30.cubin sm_30.cubin
mv ?.sm_35.cubin sm_35.cubin
mv ?.sm_52.cubin sm_52.cubin
nvdisasm sm_20.cubin -cfg > sm_20.cfg
nvdisasm sm_30.cubin -cfg > sm_30.cfg
nvdisasm sm_35.cubin -cfg > sm_35.cfg
nvdisasm sm_52.cubin -cfg > sm_52.cfg
xdot sm_20.cfg &
xdot sm_30.cfg &
xdot sm_35.cfg &
xdot sm_52.cfg &
rm -rf *.cubin
