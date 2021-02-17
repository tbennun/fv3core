#!/bin/bash

git clone git@github.com:VulcanClimateModeling/daint_venv.git
cd daint_venv
./install.sh test_ve
source test_ve/bin/activate
export MODULEPATH=${MODULEPATH}:/project/s1053/install/modulefiles
module load ccache/4.2
cd ..
# create_ccache /scratch/snx3000/tobwi/cache_test/ccache
source /project/s1053/ccache_fv3core/$experiment/ccache/bin/activate_ccache
export CCACHE_BASEDIR=`pwd`
$ROOT_DIR/examples/standalone/benchmarks/run_on_daint.sh 1 6 $backend . $data_path