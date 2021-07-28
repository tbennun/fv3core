#!/bin/bash
if [[ "${NODE_NAME}" == *"daint"* ]] ; then source ~/.bashrc ; fi

module load cray-python
rm GT4PY_VERSION.txt
echo "v30" > GT4PY_VERSION.txt
export GT4PY_VERSION="v30"
export VIRTUALENV=`pwd`/venv

make update_submodules_venv
cd external/daint_venv/
./install.sh ../../venv
source ../../venv/bin/activate
echo "1"
pip list
python -m gt4py.gt_src_manager install -m 2
cd ../..
pip install external/fv3gfs-util/
echo "2"
pip list
pip install --find-links=/project/s1053/install/wheeldir -c constraints.txt -r requirements/requirements_daint.txt
echo "3"
pip list
pip install -e .