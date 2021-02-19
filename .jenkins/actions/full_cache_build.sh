#!/bin/bash
set -e -x
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"

$ROOT_DIR/examples/standalone/benchmarks/run_with_ccache.sh 2 6 $backend . $data_path
cp -r .gt_cache_00000* /scratch/snx3000/tobwi/store_gt_caches/