

#  1.  An introduction to MR-TOD-net

MR-TOD is a deep learning network designed to predict the voxel-wise track orientation distribution (TOD) from diffusion MRI signals. The network is based on a previously published network that predicts fiber orientation distributions (FODs) from diffusion MRI signals (Lin et al. Med. Phys. 2019 46(7):3101-3116. doi: 10.1002/mp.13555). MR-TOD was trained using diffusion MRI data acquired from post-mortem mouse brains and TOD data obtained from viral tracer streamline data in the Allen mouse brain connectivity atlas (https://connectivity.brain-map.org). Details on the network can be found in our manuscript on biorxiv (doi: https://doi.org/10.1101/2022.06.02.492838). 

### The workflow of MR-TOD-net
![](https://github.com/liangzifei/MR-TOD-net/blob/main/Images/MRTod_flow.jpg) .

Fig. 1: The workflow of MR-TOD-Net. The basic network was trained using co-registered 3D Diffusion MRI and TOD map from Allen mouse connectivity project. The TOD map was generated from Allen tracer streamlines from more than 2700 subjects (https://connectivity.brain-map.org/).

# 2. How to use MR-TOD-Net?
1. Predict the TOD from diffusion MRI without training:

Use our pre-trained neural networks to predict TOD from RAW diffusion MRI data, the diffusion gradient table can be found under ./Data acquisition.

Our trained model uploaded and can be found at https://osf.io/hda8r/

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
MR-TOD takes co-registered Diffusion MRI and target TOD map for training and testing. You can find our 3D diffusion MRI(https://osf.io/hda8r/) and matched TOD link (https://osf.io/3c2xq/). Details on MRI data acquistion, source of Tracer steramlines data, and co-registration steps can be found in our manuscript ([doi: https://doi.org/10.1101/2022.06.02.492838](https://www.biorxiv.org/content/10.1101/2022.06.02.492838v1.full.pdf+html)). If you plan to use our trained networks without modifications, it is important to use the same image acquisition protocols.

The figure below shows three examples of coregistered MRI_TOD. Here, we overlap the TOD obtained from Allen tracer streamlines on our scanned mouse FA map.
![](https://github.com/liangzifei/MR-TOD-net/blob/main/Images/TOD_overlap6.png)

## 4.2 network training
Once co-registered dMRI and TOD data are ready, please run the following steps.
### 1) Run training generation code under ./Matlab-CODE: Generate_train.m
Within the code, the user can modify the following part.
```
files = dir('.\Matlab-CODE\Kim');     <--- directory of subject data used for training.
dwi5000 = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\rawdata1.mif']).  <--- diffusion data used for training.
tod_img = read_mrtrix([folder_tod,'\tod_fromtckTODp60_lmax6_to',folder_list(sample_img).name,'.mif']);   <--- target TOD data used for training.
```
The code will generate and save .npy file for the next step training.

### 2) Run deep learning code under ./CODE to train the neural network: train.py
#### We used the pycharm platform to run python code. 

Within the code, the user can modify the following part as their own paths.
```
 parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                        default='./Matlab-CODE/' ,
                    help='Path for input images').              <--- The path that saved last step .npy training diffusion data

parser.add_argument('-tgt', '--tgt_dir', action='store', dest='tgt_dir',
                        default='./Matlab-CODE/',
                        help='Path for input images').          <--- The path that saved last step .npy training TOD data
                    
parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./CODE/output/' ,
                    help='Path for Output images')              <--- The path that saved output data
    
parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./CODE/model/' ,
                    help='Path for model')                      <--- The path that saved training model
```
Some additional parameters that can be updated as the following:

```
parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=870000,
                    help='Number of Images', type= int).        <--- The number of training samples, should match the last step produced .npy file.
                    
parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.95,
                    help='Ratio of train and test Images', type=float).   <--- The number of samples for validation, here 5% for velidatation.
                    
image_shape = (3,3,3, 60)                                      <--- The number of gradient direction.
```
After training the saved model will be saved in ./model and our pre-trained model is uploaded online https://osf.io/hda8r/

## 4.3 network testing
### 1) Run testing generation code under ./Matlab-CODE: Generate_test.m
Within the code, the user can modify the following part as their own paths.
```
files = dir('.\Matlab-CODE\Kim\tJN*');
dwi5000 = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\rawdata1.mif'])
writeNPY(data,'R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn\test_input.npy');
```
As the PC memory sometimes limited, process all voxels in oneset might be impossible. Here we pick one slab containing only 60 slices one time.
Users can update the code according their own PC memory.
```
repeat = 1;    <--- repeat is the slab number
sample_img = 1; <--- this is the subject ID
slice0 =(repeat-1)*60+41;   <--- we start from slice 41, as those start slices prior than 40 are mostly background.
for slice=slice0:slice0+60%. <--- we pick 60 slice voxels for one time. 
```

### 2) Run network predication code under ./CODE: test.py
Users can modify the following as their own paths:
```
  parser.add_argument('-ihr', '--input_hig_res', action='store', dest='input_hig_res',
                        default='./Matlab-CODE/',
                    help='Path for input images Hig resolution')
                    
  parser.add_argument('-ilr', '--input_low_res', action='store', dest='input_low_res',
                        default='./Matlab-CODE/',
                    help='Path for input images Low resolution')
                    
  parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./CODE/output/',
                    help='Path for Output images')
    
  parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./CODE/model/gen_modelTOD700_28channelKimBatch512.h5' ,
                    help='Path for model')
```

### 3) Run the restruction code under ./Matlab-CODE: Generate_test_Recon.m
As the network is voxel-wised processing, it is required to reconstruct the entire 3d TOD map by packing all the voxels to the original subject space.
Users can modify their own paths:
```
files = dir('.\Matlab-CODE\Kim\tJN*');
% Get a logical vector that tells which is a directory.
folder_list = files;
folder_dwi =['.\Matlab-CODE\Kim\'];
```

# License
MIT License

Copyright (c) 2021 liangzifei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
