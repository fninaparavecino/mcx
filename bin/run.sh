#!/bin/bash
APP="mcx"
./$APP
cuobjdump ./$APP -xelf all
nvdisasm mcx_core.sm_20.cubin -cfg > sm_20.cfg
nvdisasm mcx_core.sm_35.cubin -cfg > sm_35.cfg
nvdisasm mcx_core.sm_52.cubin -cfg > sm_52.cfg
xdot sm_20.cfg &
xdot sm_35.cfg &
xdot sm_52.cfg &
rm -rf *.cubin
