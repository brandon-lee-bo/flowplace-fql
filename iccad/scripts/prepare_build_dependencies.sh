#!/bin/bash
# Dependency staging only; compilation and all EDA runs belong in Slurm.
# Run from the experiment root on the host with the installed Boost-devel RPM.
# This never changes shared/system software.
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
mkdir -p iccad/assets/deps/include iccad/assets/deps/lib iccad/assets/deps/rpmroot
cp -a /usr/include/boost iccad/assets/deps/include/
cp -a /usr/lib64/libboost_graph.so /usr/lib64/libboost_graph.so.1.75.0 \
      /usr/lib64/libboost_regex.so /usr/lib64/libboost_regex.so.1.75.0 \
      /usr/lib64/libpython3.9.so.1.0 iccad/assets/deps/lib/
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
dnf --disablerepo='*' --enablerepo=appstream --enablerepo=baseos download \
    --destdir=iccad/assets/deps python3-devel-3.9.25-7.el9_8.3.x86_64
rpm2cpio iccad/assets/deps/python3-devel-3.9.25-7.el9_8.3.x86_64.rpm |
    cpio -idm --quiet -D iccad/assets/deps/rpmroot
cp -a /usr/include/python3.9/pyconfig-64.h iccad/assets/deps/rpmroot/usr/include/python3.9/
