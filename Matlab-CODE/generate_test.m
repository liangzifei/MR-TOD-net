function flag = generate_test(repeat, sample_img)
%%
%loop several times for the data generate.
%%
files = dir('R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim\tJN*');
% Get a logical vector that tells which is a directory.
% dirFlags = [files.isdir];
% Extract only those that are directories.
folder_list = files;
% folder_list = dir(['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kaffman_Exp55\1*']);
folder_dwi =['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim\'];
% folder_tod = ['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kaffman_Exp55\Z_todVersions\TOD_fromTCK_lmax6\'];
%% start loop %%%%%%%%%%%%%%%%
halfsize_input = 1;
stride = 1;
%%
slice0 =(repeat-1)*60+41;     count=0;
%%
dwi5000 = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\rawdata1.mif'])
%     fod_img = read_mrtrix([folder_dwi,'fod-z',file_list(sample_img).name(1:2),'.mif']);
%     tod_img = read_mrtrix([folder_dwi,'tod_sample150normed',file_list(sample_img).name(1:2),'sh.mif']);
% tod_img = read_mrtrix([folder_tod,'tod_fromtckTODp60_lmax6_to',folder_list(sample_img).name,'.mif']);
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
% tod_data(isnan(tod_data))=0;
%% end normalize.
% sum_tod4 = sum(tod_data,4); %sum_tod4(sum_tod4<10) = 0;
% mask_data = logical(sum_tod4); fa_img.img(isnan(fa_img.img))=0;
% mask_fa = fa_img.img; %mask_fa(mask_fa<0.01) = 0;
% mask_fa = logical(mask_fa);

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
    writeNPY(data,'R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn\test_input.npy');
%     label = permute(label,[5,1,2,3,4]);
%     writeNPY(label,'R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn\test_output.npy');
    flag = 1
else
    flag =0
end