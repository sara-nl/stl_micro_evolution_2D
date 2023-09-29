# Spatiotemporal Learning of Microstructure Evolution
Spatiotemporal ML techniques to accelerate learning of microstructure and enable computationally affordable evolution predictions.

Load the relevant modules:
```
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 torchvision/0.13.1-foss-2022a-CUDA-11.7.0
module load VTK/9.2.0.rc2-foss-2022a h5py/3.7.0-foss-2022a Pillow/9.1.1-GCCcore-11.3.0 OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
```

## Installation
```
cd OpenSTL
pip install -r requirements.txt
python setup.py develop
```

OpenSTL is a comprehensive benchmark for spatio-temporal predictive learning from https://github.com/chengtan9907/OpenSTL/ .

Method added based on Unet and Swin transformers: VT Unet
original Pytorch implementation: https://github.com/himashi92/VT-UNet for segmentation tasks. Reporposed for prediction tasks. 
original paper: https://arxiv.org/pdf/2111.13300.pdf 

Microstructure datasets added:
- Phase Field  https://archive.materialscloud.org/record/2022.156
- Monte Carlo Potts




