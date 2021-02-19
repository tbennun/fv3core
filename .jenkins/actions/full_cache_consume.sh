#!/bin/bash
set -e -x
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"

cp -r /scratch/snx3000/tobwi/store_gt_caches/.gt_cache_00000* .
find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/tobwi\/sbox\/consume\/fv3core|$(pwd)|g" {} +
$ROOT_DIR/examples/standalone/benchmarks/run_with_ccache.sh 2 6 $backend . $data_path