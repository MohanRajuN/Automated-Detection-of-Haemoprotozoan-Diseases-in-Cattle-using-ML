clc
clear all
close all
addpath(genpath('.'))
%% Loading Feature values

load Fea
for N1=1:2%Iteration
    Tot_imag=size(Fea{1},1);
    ran_val=randperm(Tot_imag);
    Train=[];
    Test=[];Train_img=[];Test_img=[];
    %% Training Seperation
    Train_len=round(Tot_imag*0.6);
    Test_len=Tot_imag-Train_len;
    %Training and testing seperation
    for i= 1:length(Fea)
        Train=[Train ;Fea{i}(ran_val(1:Train_len),:) ];
        Test=[Test; Fea{i}(ran_val(Train_len+1:end),:)];
    end
    
    Train(isinf(Train))=0;
    Test(isinf(Test))=0;
    Train(isnan(Train))=0;
    Test(isnan(Test))=0;
    %Setting target values
    Train_tar=repmat([1:length(Fea)],[Train_len,1]);
    Test_tar=repmat([1:length(Fea)],[Test_len,1]);
    Train_tar=Train_tar(:);
    Test_tar=Test_tar(:);
    
    %% Infinite feature selection
    %     alpha = 0.5;
    %     sup = 1;
    %     [Feat_Index, w] = infFS( Train , Train_tar, alpha , sup , 0 );
    %     k=round(size(Feat_Index',1).*0.3);
    %     Train=Train(:,Feat_Index(1:k));
    %     Test=Test(:,Feat_Index(1:k));
    %% Mutual Information based Feature selection
%     [ Feat_Index , w] = mutInfFS( Train, Train_tar, length(Train) );
    %     k=round(size(Feat_Index,1).*0.3);
    %     Train=Train(:,Feat_Index(1:k));
    %     Test=Test(:,Feat_Index(1:k));
    
    %% Correlation based feature selection
    %     Feat_Index = corrSel(Train,Train_tar);
    %     k=round(size(Feat_Index',1).*0.3);
    %     Train=Train(:,Feat_Index(1:k));
    %     Test=Test(:,Feat_Index(1:k));
    %% Relief
    [Feat_Index, w] = reliefF( Train , Train_tar, 20  );
    k=round(size(Feat_Index',1).*0.3);
    Train=Train(:,Feat_Index(1:k));
    Test=Test(:,Feat_Index(1:k));
    %% MSVM
    svmParams = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto', 'Standardize', 1);
    SVMStruct = fitcecoc(Train, Train_tar ,'Learners', svmParams);
    label=predict(SVMStruct,Test);
    [Svm_Parameter(N1)]=Finding_parameter1(Test_tar,label);
    
    %% RF Decision Tree
    Mdl = TreeBagger(5,Train,Train_tar);
    label= predict(Mdl,Test);
    label=double(string(label));
    [RF_Parameter_5(N1)]=Finding_parameter1(Test_tar,double(string(label)));
    %% DNN
    Test_tar(Test_tar==3)=4;
    Train_tar1(Train_tar==3)=4;
    Train_tar1(Train_tar==2)=2;
    Train_tar1(Train_tar==1)=1;
    deepnet=Deep_Train((Train'),de2bi(Train_tar1')');
    label=Deep_Predict(deepnet,(Test'));
    [Deep_Parameter1(N1)]=Finding_parameter1(Test_tar,label);
    %% KNN
    Knn_b = fitcknn(Train,Train_tar);
    label= predict(Knn_b,Test);
    [Knn_Parameter(N1)]=Finding_parameter1(Test_tar,label);
    
    

%% Results
Result_Acc=[ mean([Deep_Parameter1.Accuracy]); mean([Svm_Parameter.Accuracy]);...
    ;mean([RF_Parameter_5.Accuracy]); mean([Knn_Parameter.Accuracy])];
Result_Sen=[mean([Deep_Parameter1.sensitivity]); mean([Svm_Parameter.sensitivity]);...
    ;mean([RF_Parameter_5.sensitivity]); mean([Knn_Parameter.sensitivity])];
Result_Spc=[mean([Deep_Parameter1.specificity]); mean([Svm_Parameter.specificity]);...
    ;mean([RF_Parameter_5.specificity]); mean([Knn_Parameter.specificity])];
Result_Mcc=[mean([Deep_Parameter1.MCC]); mean([Svm_Parameter.MCC]);...
    ;mean([RF_Parameter_5.MCC]); mean([Knn_Parameter.MCC])];
Result_Fscore=[mean([Deep_Parameter1.F1_score]); mean([Svm_Parameter.F1_score]);...
    ;mean([RF_Parameter_5.F1_score]); mean([Knn_Parameter.F1_score])];
Result=[Result_Acc Result_Sen Result_Spc Result_Mcc  Result_Fscore];
Result(isnan(Result))=0;
Classifier={ 'DNN','MSVM','RF','KNN'};
Parameters={'Accuracy';'sensitivity';'specificity';'MCC';'F_Score'};
%% Tables
Results=array2table((Result*100),'VariableNames',Parameters','RowNames', Classifier');
disp(Results)
end
