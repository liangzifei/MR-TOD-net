

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
## 4.1 Data Preparation
MR-TOD takes co-registered Diffusion MRI and target TOD map for training and testing. You can find our 3D diffusion MRI(https://osf.io/hda8r/) and matched TOD link (https://osf.io/3c2xq/). Details on MRI data acquistion, source of Tracer steramlines data, and co-registration steps can be found in our manuscript (doi: https://doi.org/10.1101/2022.06.02.492838). If you plan to use our trained networks without modifications, it is important to use the same image acquisition protocols.

The figure below gives examples of coregistered MRI_TOD. Here, we overlap the TOD obtained from Allen tracer streamlines on our subjects.


## 4.2 network training
Once co-registered MRI and target histological data are ready, use demo_trainingPrep.m in Matlab to prepare training samples for the next step.
### 1) Run training generation code under Matlab-CODE: Generate_train.m
Within the code the user need to modify the following part.
```
files = dir('R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim');     <--- directory of subject data used for training.
dwi5000 = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\rawdata1.mif']).  <--- diffusion data used for training.
tod_img = read_mrtrix([folder_tod,'\tod_fromtckTODp60_lmax6_to',folder_list(sample_img).name,'.mif']);   <--- target TOD data used for training.
```
The code will generate and save .npy file for the next step training.

### 2) Run deep learning code under CODE to train the neural network: Train.py

Within the code the user need to modify the following part as their own folders.
```
 parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                        default='R:/zhangj18lab/zhangj18labspace/Zifei_Data/MouseHuman_proj/DeepNet_Learn/' ,
                    help='Path for input images')

parser.add_argument('-tgt', '--tgt_dir', action='store', dest='tgt_dir',
                        default='R:/zhangj18lab/zhangj18labspace/Zifei_Data/MouseHuman_proj/DeepNet_Learn/',
                        help='Path for input images')
                    
parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')
    
parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/' ,
                    help='Path for model')
```
Some additional parameters that can be updated as following:

```
    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=870000,
                    help='Number of Images', type= int)
                    
    parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.95,
                    help='Ratio of train and test Images', type=float)
                    
    image_shape = (3,3,3, 60)
```


## 4.3 network testing
