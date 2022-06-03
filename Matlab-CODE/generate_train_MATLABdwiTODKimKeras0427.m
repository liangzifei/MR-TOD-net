clear;close all;
files = dir('R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim');
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
folder_list = files(dirFlags); folder_list(1:2) = []; 
% folder_list = dir(['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kaffman_Exp55\1*']);
folder_dwi =['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim\'];
folder_tod = ['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim\Z_tod\'];
%% start loop %%%%%%%%%%%%%%%%
halfsize_input = 1;
stride = 3;
count=0;
% sample_num=[1,2,4,5];
for sample_img = 1:7% length(folder_list)
    %     dwi2000 = load_untouch_nii([folder_dwi,num2str(sample_img),'\rigidaffine_Lddm_dwi2000.img']);
    dwi5000 = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\rawdata1.mif'])
    %     fod_img = read_mrtrix([folder_dwi,'fod-z',file_list(sample_img).name(1:2),'.mif']);
    %     tod_img = read_mrtrix([folder_dwi,'tod_sample150normed',file_list(sample_img).name(1:2),'sh.mif']);
    tod_img = read_mrtrix([folder_tod,'\tod_fromtckTODp60_lmax6_to',folder_list(sample_img).name,'.mif']);
    %% dwi data preprocess S/S0 DATA
    dwi_data = double(dwi5000.data(:,:,:,1:62)); tod_data = double(tod_img.data(:,:,:,1:28)); clear dwi5000 tod_img;
    
    fa_img = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\warped_P60.mif']);
    mask_img = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\mask_shrink.mif']);
    
    norm = mean(dwi_data(:,:,:,1:2),4);
%     for num =1:60
%         dwi_data1(:,:,:,num) = dwi_data(:,:,:,num+2)./norm;
%     end
    
    dwi_data1 = dwi_data(:,:,:,3:end);
    dwi_data = dwi_data1./norm;
    
    clear dwi_data1;
    tod_data(isnan(tod_data))=0;
    %     norm_todmat = repmat(norm_todmat,[1,1,1,45]);
    %     tod_data = tod_data./norm_todmat;
    %% normalize data form original.
    %     S0 = dwi_data(:,:,:,1:2); S=dwi_data(:,:,:,3:end);
    %     S0 = repmat(mean(S0,4),[1,1,1,size(S,4)]);
    %     SdivS0 = S./S0; norms = sqrt(sum(SdivS0.^2,4));
    %     norms = repmat(norms,[1,1,1,30]);
    %     SdivS0= SdivS0./norms;
    %     SdivS0(isnan(SdivS0))=0;
    %% end normalize.
    sum_tod4 = sum(tod_data,4); %sum_tod4(sum_tod4<10) = 0;
    mask_data = logical(sum_tod4).*logical(mask_img.data); fa_img.data(isnan(fa_img.data))=0;
    mask_fa = fa_img.data; mask_fa(mask_fa<0.01) = 0; mask_fa = logical(mask_fa);
    
    mask_data = mask_data.*mask_fa;
    %%
    [hei,wid,C,channel]=size(dwi_data);
    %% loop count samples %%%%%%%%%%%%%%%%%%%%%%%%%
    for slice=1+halfsize_input : stride : C-halfsize_input
        for x = 1+halfsize_input : stride : hei-halfsize_input
            for y = 1+halfsize_input :stride : wid-halfsize_input
                % include all MRI contrasts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if(mask_data(x,y,slice)~=0&count<2000000)
                    subim_input = dwi_data(x-halfsize_input : x+halfsize_input,...
                        y-halfsize_input : y+halfsize_input,slice-halfsize_input : slice+halfsize_input,[1:60]);
%                     sum_norm = sum(subim_input,4);
%                     subim_input = subim_input./sum_norm;
                    subim_input = subim_input./max(subim_input(:));
                    %% only diffusion MRI %%%%%%%%%%%%%%%%%%%%%%%%%
                    subim_label = tod_data(x ,y, slice,:); subim_label = subim_label./max(abs(subim_label(:)));
                    
                    count=count+1;
                    data(:, :, :, :, count) = subim_input;
                    label(:, :, :, :,count) = subim_label;
                    
                    disp(['...',num2str(x),'...',num2str(y),'...',num2str(slice),'...']);
                else
                    continue;
                end
            end
        end
    end
end

order = randperm(count);
data = data(:, :, :,:,order);
label = label(:, :,:, :, order);
%%
%%
data = permute(data,[5,1,2,3,4]);
writeNPY(data,'input.npy');
label = permute(label,[5,1,2,3,4]);% label = label;
writeNPY(label,'output.npy');
% save traindataJG_allMRIsTOD.mat data label -v7.3;