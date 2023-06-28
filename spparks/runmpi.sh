#!/bin/bash
#SBATCH --job-name=get_data_spparks            # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --partition=thin

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0
module load spparks/16Jan23-foss-2022a


SPPARKS="${HOME}/esa/IN100_SLM_AI_Training_Set_II/spparks" #make sure that in.potts_am_IN100_3d and IN100_3d.init are inside here

python init_config.py --working_dir ${SPPARKS}

config_file="${SPPARKS}/config_file"

while IFS= read -r line; do
    path="$(echo -e "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    echo "Processing: $path/in.potts_am_IN100_3d"
    mpirun -np 32 spk_mpi -echo log < ${SPPARKS}/${path}/in.potts_am_IN100_3d

done < "$config_file"
