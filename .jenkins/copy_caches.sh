#!/bin/bash
set -e -x
SANITIZED_BACKEND=$1
EXPERIMENT_NAME=$2
GT4PY_VERSION=$3

if [ ! -d $(pwd)/.gt_cache_000000 ]; then
    version_file=/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$SANITIZED_BACKEND/GT4PY_VERSION.txt
    if [ -f ${version_file} ]; then
	version=`cat ${version_file}`
    else
	version=""
    fi
    if [ "$version" == "$GT4PY_VERSION" ]; then
        cp -r /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$SANITIZED_BACKEND/.gt_cache_0000* .
        find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/fv3core-cache-setup\/backend\/$SANITIZED_BACKEND\/experiment\/$EXPNAME\/slave\/daint_submit|$(pwd)|g" {} +
    fi
fi