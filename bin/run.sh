#!/bin/bash
APP="mcx"
cuobjdump $APP -xelf mcx_core.sm_52.cubin
nvdisasm mcx_core.sm_52.cubin -cfg > sm_52_70.dot
dot -Tps -o 980Ti/graph_sm_52_70.ps sm_52_70.dot
# xdot sm_52.dot &
rm -rf *.cubin
