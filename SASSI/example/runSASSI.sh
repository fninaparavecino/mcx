#!/bin/bash
APP="kernelMatrixMul"
make cfg
./$APP
mv sassi-cfg.dot sassi-cfg-980Ti.dot
xdot sassi-cfg-980Ti.dot &
