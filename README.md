

#  1.  An introduction to MR-TOD-net

MR-TOD is a deep learning network designed to predict the voxel-wise track orientation distribution (TOD) from diffusion MRI signals. The network is based on a previously published network that predicts fiber orientation distributions (FODs) from diffusion MRI signals (Lin et al. Med. Phys. 2019 46(7):3101-3116. doi: 10.1002/mp.13555). MR-TOD was trained using diffusion MRI data acquired from post-mortem mouse brains and TOD data obtained from viral tracer streamline data in the Allen mouse brain connectivity atlas (https://connectivity.brain-map.org). Details on the network can be found in our manuscript on biorxiv (doi: https://doi.org/10.1101/2022.06.02.492838). 

### The workflow of MR-TOD-net

Fig. 1: The workflow of MR-TOD-Net. The basic network was trained using co-registered 3D Diffusion MRI and TOD map from Allen mouse connectivity project. The TOD map was generated from Allen tracer streamlines from more than 2700 subjects (https://connectivity.brain-map.org/).

# 2. How to use MR-TOD-Net?
1. Predict the TOD from diffusion MRI without training:

Use our trained deep neural networks to predict FOD from RAW diffusion MRI signals, by input the same graindient scanned data.

Our trained model uploaded as https://osf.io/hda8r/

2. Train you own neural network:

The spacial matched dMRI and TOD data is uploaded. It is provided to train new neural networks. 

The target TOD is provided as https://osf.io/3c2xq/

The matched dMRI data is provided as https://osf.io/hda8r/

## Tractography streamlines resources
The original fibers are downloaded from Allen connectivity program (https://connectivity.brain-map.org/static/brainexplorer)

Our whole-brain streamline fibers was uploaded as https://osf.io/m98wn/
# 3. Requirements


Windows 10
Matlab version > 2019b
Deep learning toolbox. https://www.mathworks.com/products/deep-learning.html
nifti toolbox https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
CUDA
Pytorch (Python 3.6)
# 4. Procedure
