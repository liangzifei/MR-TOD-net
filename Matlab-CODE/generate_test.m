% generate test diffusin MRI data for TOD prediction uisng trained neural network
clc;clear;
% because of the memory limitation, we segment the data into several slabs to run.
% Users can update the code to remove the slab segmentation. 
repeat = 1; %repeat is the slab number
sample_img = 1;
function flag = generate_test(repeat, sample_img)
%%
%loop several times for the data generate.
%%
files = dir('.\Matlab-CODE\Kim\tJN*');
% Get a logical vector that tells which is a directory.
% dirFlags = [files.isdir];
% Extract only those that are directories.
folder_list = files;
folder_dwi =['.\Matlab-CODE\Kim\'];
% folder_tod = ['.\TOD_fromTCK_lmax6\'];
%% start loop %%%%%%%%%%%%%%%%
halfsize_input = 1;
stride = 1;
%%
slice0 =(repeat-1)*60+41;     count=0;
%%
dwi5000 = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\rawdata1.mif'])
%% dwi data preprocess S/S0 DATA
dwi_data = double(dwi5000.data(:,:,:,1:62)); %tod_data = double(tod_img.data(:,:,:,1:28)); 
clear dwi500;
% fa_img = load_untouch_nii([folder_dwi,folder_list(sample_img).name,'\FA.nii']);
mask_img = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\mask.mif']);
norm = mean(dwi_data(:,:,:,1:2),4);
%     for num =1:60
%         dwi_data1(:,:,:,num) = dwi_data(:,:,:,num+2)./norm;
%     end
dwi_data1 = dwi_data(:,:,:,3:end);
dwi_data = dwi_data1./norm;

clear dwi_data1;

mask_data = logical(mask_img.data); mask_data(isnan(mask_data))=0;
%%
[hei,wid,C,channel]=size(dwi_data);
%% loop count samples %%%%%%%%%%%%%%%%%%%%%%%%%
for slice=slice0:slice0+60%C-1%1+halfsize_input : stride : C-halfsize_input
    for x = 1+halfsize_input : stride : hei-halfsize_input
        for y = 1+halfsize_input :stride : wid-halfsize_input
            if mask_data(x,y,slice)
                subim_input = dwi_data(x-halfsize_input : x+halfsize_input,...
                    y-halfsize_input : y+halfsize_input,slice-halfsize_input : slice+halfsize_input,[1:60]);
                %% only diffusion MRI %%%%%%%%%%%%%%%%%%%%%%%%%
                subim_input = subim_input./max(subim_input(:));
                count=count+1;
%                 subim_label = tod_data(x ,y, slice,:);
                data(:, :, :, :, count) = subim_input;
%                 label(:, :, :, :,count) = subim_label;
                disp(['...',num2str(x),'...',num2str(y),'...',num2str(slice),'...']);
            else
            end
        end
    end
end
if (exist('data')==1)
    data= permute(data,[5,1,2,3,4]);
    writeNPY(data,'.\Matlab-CODE\test_input.npy');
%     label = permute(label,[5,1,2,3,4]);
%     writeNPY(label,'.\Matlab-CODE\test_output.npy');
    flag = 1
else
    flag =0
end
