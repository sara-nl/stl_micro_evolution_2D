#!/bin/bash
#SBATCH --job-name=get_data_spparks            # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --partition=thin

module purge
module load 2022
module load spparks/16Jan23-foss-2022a


SPPARKS="${HOME}/esa/ml-materials-engineering/spparks" #define your path
config_file="${SPPARKS}/config_file"

while IFS= read -r line; do
    path="$(echo -e "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    echo "Processing: $path/in.potts_am_IN100_3d"
    mpirun -np 32 spk_mpi -echo log < ${SPPARKS}/${path}/in.potts_am_IN100_3d

done < "$config_file"
