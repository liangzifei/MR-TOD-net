function flag = generate_test_Recon(repeat, sample_img)
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
count =0;
%%
slice0 =(repeat-1)*60+41;
%%

tod_img = read_mrtrix(['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Kim\Z_tod',...
    '\tod_fromtckTODp60_lmax6_to',folder_list(sample_img).name,'.mif']);
%%
YPred = readNPY(['K:\SRCNN_deltADC\Pytorch_code\Keras-SRGAN\',...
    'SR3d\Keras-Resnet-dwi2fod\output\Test_output_fod.npy']);
% label = readNPY(['R:\zhangj18lab\zhangj18labspace\Zifei_Data\',...
%     'MouseHuman_proj\DeepNet_Learn\test_output.npy']);

tod_img.data(isnan(tod_img.data)) = 0;
tod_data = double(tod_img.data(:,:,:,1:28));
%             mask_data = logical(sum(fod_img.data,4));
%             mask_img = read_mrtrix(['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\Fiber_validate\20210107DWI_match_TOD\',...
%                 'mask_',file_list(sample_img).name(1:2),'.mif']);
%             mask_data = logical(mask_img.data);
% fa_img = load_untouch_nii([folder_dwi,folder_list(sample_img).name,'\FA.nii']);
mask_img = read_mrtrix([folder_dwi,folder_list(sample_img).name,'\mask.mif']);
% sum_tod4 = sum(tod_data,4); %sum_tod4(sum_tod4<10) = 0;
% mask_data = logical(sum_tod4); fa_img.img(isnan(fa_img.img))=0;
% mask_fa = fa_img.img; %mask_fa(mask_fa<0.01) = 0;
% mask_fa = logical(mask_fa);

mask_data = logical(mask_img.data); mask_data(isnan(mask_data)) = 0;

tod_data = zeros(size(tod_img.data)); %ref_data = tod_data;
[hei,wid,C,channel]=size(tod_data);
%% loop count samples %%%%%%%%%%%%%%%%%%%%%%%%%
for slice=slice0:slice0+60%C-1%1+halfsize_input : stride : C-halfsize_input
    for x = 1+halfsize_input : stride : hei-halfsize_input
        for y = 1+halfsize_input :stride : wid-halfsize_input
            %% only diffusion MRI %%%%%%%%%%%%%%%%%%%%%%%%%
            if(mask_data(x,y,slice))
                count=count+1;
                tod_data(x ,y, slice,1:28) = YPred(count,:,:,:,:);
%                 ref_data(x,y,slice,1:28)= label(count,:,:,:,:);
                disp(['...',num2str(x),'...',num2str(y),'...',num2str(slice),'...']);
            else
            end
        end
    end
end
tod_img.data = tod_data;
write_mrtrix(tod_img,['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn\Recon_Tod',num2str(slice0),'-',num2str(slice0+60),'_t35b.mif']);
% tod_img.data = ref_data;
% write_mrtrix(tod_img,['R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn\Ref_TOD',num2str(slice),'-',num2str(slice+20),'_702FC.mif']);
flag = 1