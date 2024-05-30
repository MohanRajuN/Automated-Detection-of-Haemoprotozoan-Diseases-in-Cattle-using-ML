clc
clear all
close all
Get_data='babesiosis';
Get_data1=dir(Get_data);
Get_data1(1:2)=[];
addpath(genpath('.'))
mkdir([Get_data '_out'])
for N=2:length(Get_data1)
    %% Read images
    Read_path= [Get_data '\' Get_data1(N).name];
    Img_read=((imread(Read_path)));
    figure,imshow(Img_read),title('Input')
    %% Image normalization
    Img_norm=Image_Normilization(double(Img_read));
    figure,imshow(Img_norm),title('Normalization')
    %% Edge Detection
    Img_edge=edge(rgb2gray(Img_norm),'canny');
    figure,imshow(Img_edge),title('Edge detection')
    %% Remove smallest components
    y1 = bwareaopen( Img_edge , 50 ) ;
    %% Circular hough Transform
    [centers, radii, metric] = imfindcircles(y1,[10 30]);
    RGB = insertShape(Img_read,'circle',[centers(1,:) radii(1)],'LineWidth',5);
%     mkdir([Get_data '_out' '\' Get_data1(N).name])
    
    for i=1:length(radii)
        vessel=Img_read;
        imshow( vessel)
        t = 0:pi/20:2*pi;
        xc=centers(i,1); % point around which I want to extract/crop image
        yc=centers(i,2);
        r=radii(i);   %Radium of circular region of interest
        xcc = r*cos(t)+xc;
        ycc =  r*sin(t)+yc;
        roimaskcc = poly2mask(double(xcc),double(ycc), size(vessel,1),size(vessel,2));
        P=regionprops(roimaskcc);
        A=imcrop(Img_norm.*1,P.BoundingBox);
        imwrite(A,[Get_data '_out' '\' Get_data1(N).name '\'   num2str(i) '_' Get_data1(N).name])
    end
    viscircles(centers, radii,'Color','b');
    close all
end