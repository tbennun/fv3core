#!/bin/bash
set -e -x
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"

git clone git@github.com:VulcanClimateModeling/daint_venv.git
cd daint_venv
./install.sh test_ve
source test_ve/bin/activate
cd ..

export MODULEPATH=${MODULEPATH}:/project/s1053/install/modulefiles
module load ccache/4.2
source /project/s1053/ccache_fv3core/$experiment/ccache/bin/activate_ccache
export CCACHE_BASEDIR=`pwd`
$ROOT_DIR/examples/standalone/benchmarks/run_with_ccache.sh 2 6 $backend . $data_path