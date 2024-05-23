#!/bin/bash
#SBATCH --job-name=train_pf              # Job name
#SBATCH --nodes=1  
#SBATCH --ntasks=1                      # Run on a single CPU
#SBATCH --time=3-00:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=monica.rotulo@surf.nl

module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 torchvision/0.13.1-foss-2022a-CUDA-11.7.0
module load h5py/3.7.0-foss-2022a Pillow/9.1.1-GCCcore-11.3.0 OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib

source ../.venv/openstl/bin/activate

echo  "loaded"

# train on mmnist
#python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta

# train on pf
#python tools/train.py -d pf -e 400 --lr 1e-4 -c configs/pf/predrnn/PredRNN.py --ex_name pf_predrnn_400_metrics

#python tools/train.py -d pf -e 400 --lr 1e-3 -c configs/pf/SimVP_gSTA.py --ex_name pf_simvp

#python tools/train.py -d pf -e 400 -c configs/pf/ConvLSTM.py --ex_name pf_convlstm

python tools/train.py -d pf -e 400 -c configs/pf/VTUNet.py --ex_name pf_swin --pre_seq_length 12 --aft_seq_length 12

# test on pf 
#python tools/test.py -d pf -c configs/pf/predrnn/PredRNN.py --ex_name pf_predrnn

# visualize 
#python tools/visualize/vis_video -d pf -w /home/monicar/predictive_zoo/OpenSTL/work_dirs/pf_predrnn
