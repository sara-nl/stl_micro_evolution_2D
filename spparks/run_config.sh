#!/bin/bash
#SBATCH --job-name=write_config            # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --partition=rome

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0

SPPARKS="${HOME}/esa/IN100_SLM_AI_Training_Set_II/spparks"
OUTPUT="${SPPARKS}/$(date +%Y-%m-%d_%H-%M-%S)_${SLURM_JOBID}"
mkdir -p $OUTPUT

# copy relevant files 
cp $HOME/esa/ml-materials-engineering/spparks/small_params.yaml "$TMPDIR"

python config_generator.py --yaml_file "$TMPDIR"/small_params.yaml --output_dir "$TMPDIR"

cp "$TMPDIR"/small_params.yaml $OUTPUT
cp -r "$TMPDIR"/config_file_* $OUTPUT
