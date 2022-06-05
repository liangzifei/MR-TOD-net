
# MR-TOD-net
MR-TOD is a deep learning network designed to predict the voxel-wise track orientation distribution (TOD) from diffusion MRI signals. The network is based on a previously published network that predicts fiber orientation distributions (FODs) from diffusion MRI signals (Lin et al. Med. Phys. 2019 46(7):3101-3116. doi: 10.1002/mp.13555). MR-TOD was trained using diffusion MRI data acquired from post-mortem mouse brains and TOD data obtained from viral tracer streamline data in the Allen mouse brain connectivity atlas (https://connectivity.brain-map.org). Details on the network can be found in our manuscript on biorxiv (doi: https://doi.org/10.1101/2022.06.02.492838). 

# MR-TOD-net
MR-TOD is to predict the voxel_wise axonal orientations distribution(TOD) from RAW diffusion MRI signals using deep learning

## How to use MR-TOD-Net?
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
