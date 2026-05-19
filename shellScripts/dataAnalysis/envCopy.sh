#!/bin/bash
module load miniconda3


# name of original env

SRC_ENV="motionclip"



# name of new env

DST_ENV="motionclip_clone"



echo "Activating source env..."

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate $SRC_ENV



echo "Exporting environment..."

conda env export > ${SRC_ENV}_full.yml



echo "Creating new environment..."

conda env create -f ${SRC_ENV}_full.yml -n $DST_ENV



echo "Activating new environment..."

conda activate $DST_ENV



echo "Verifying..."

python -c "import sys, numpy; print('Python:', sys.executable); print('NumPy:', numpy.__file__, numpy.__version__)"



echo "Done. New env: $DST_ENV"
