#!/bin/bash
APP="mcx"
../../bin/$APP -A -g 10 -n 1e2 -f qtest.inp -s qtest -r 1 -a 0 -b 0 -G 1
cuobjdump ../../bin/$APP -xelf all
mv ?.sm_20.cubin sm_20.cubin
mv ?.sm_35.cubin sm_35.cubin
mv ?.sm_52.cubin sm_52.cubin
nvdisasm sm_20.cubin -cfg > sm_20.cfg
nvdisasm sm_35.cubin -cfg > sm_35.cfg
nvdisasm sm_52.cubin -cfg > sm_52.cfg
xdot sm_20.cfg &
xdot sm_35.cfg &
xdot sm_52.cfg &
rm -rf *.cubin
