clc
clear all
close all
%% Get the database
addpath(genpath('.'))
database='./Database/';
Get_db=dir(database);
Get_db(1:2)=[];Fea=[];
AA=1;
for N1=1:3
    Read_fol=dir([database Get_db(N1).name]);
    Read_fol(1:2)=[];Fea1=[];
    for N2=1:50
        %% Reading Images
        BW22=imresize(imread([database Get_db(N1).name '/' Read_fol((N2)).name]),[32 32]);
%         s = regionprops(logical(rgb2gray(BW22)));
%         [val,id]=max([s.Area]);
%         BW22=rgb2gray(imresize(imcrop(BW22,s(id).BoundingBox),[32 32]));
        Fea1=[Fea1;BW22(:)'];
        Train(:,:,:,AA)=BW22;
        AA=AA+1;
        close all
    end
       %% Store feature values
    Fea=[Fea;{normalize(double(Fea1))}];
    close all
end
% save Fea Fea
% save Train Train


