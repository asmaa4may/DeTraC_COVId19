function [dataset_B ] = MultiClass_ErrorCorrection(cmat, org_classNum)

%% This function uses to reassemble back each sub-classes within dataset B
% into its original class in dataset A
% for example: 
% class (COVID_19)<----------------(COVID_19_1 and COVID_19_2)

m=1;  d=2;  TP=0;   
dataset_B = zeros(org_classNum,org_classNum);

 %%
    for C=1:org_classNum
            TP=0;
             for i=m:d
                 for j=m:d
                    TP= TP + cmat(i,j);
                 end
             end
             dataset_B(C,C)=TP;
             m=m+2;
             d=d+2;
    end

     dataset_B(1,2)=cmat(1,3)+ cmat(1,4)+cmat(2,3)+cmat(2,4);
     dataset_B(1,3)=cmat(1,5)+cmat(1,6)+cmat(2,5)+cmat(2,6);
     dataset_B(2,1)= cmat(3,1)+ cmat(3,2)+cmat(4,1)+cmat(4,2);
     dataset_B(2,3)= cmat(3,5)+ cmat(3,6)+ cmat(4,5)+ cmat(4,6);
     dataset_B(3,1)=cmat(5,1)+ cmat(5,2)+ cmat(6,1)+ cmat(6,2);
     dataset_B(3,2)=cmat(5,3)+ cmat(5,4)+ cmat(6,3)+ cmat(6,4);
     
end

