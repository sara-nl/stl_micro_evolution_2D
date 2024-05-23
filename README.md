# Spatiotemporal Learning of Microstructure Evolution
This project applies spatiotemporal ML techniques for learning the microstructrure formation in materials and enable computationally affordable evolution predictions.

## Installation
Load the relevant modules:
```
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 torchvision/0.13.1-foss-2022a-CUDA-11.7.0
module load VTK/9.2.0.rc2-foss-2022a h5py/3.7.0-foss-2022a Pillow/9.1.1-GCCcore-11.3.0 OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
```
And install the required packages:
```
pip install -r requirements.txt
python setup.py develop
```

## Data 
Microstructure datasets considered:
- Phase Field:  https://archive.materialscloud.org/record/2022.156
- Monte Carlo Potts: https://doi.org/10.5281/zenodo.11209659

## Credit
This implementation alters and re-uses models from the following sources:
- OpenSTL Benchmark: https://github.com/chengtan9907/OpenSTL/ 
- VT Unet: https://github.com/himashi92/VT-UNet




