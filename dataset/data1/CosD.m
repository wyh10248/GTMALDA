function [ CosD] = cosSim( data )
%COSSIM Summary of this function goes here
%   Detailed explanation goes here
data=load('disease-lncRNA.csv'); 
rows=size(data,1);

    for i=1:rows
        
        for j=1:rows
            
            if (norm(data(i,:))*norm(data(j,:))==0)
                
                CosD(i,j)=0;
                
            else
                CosD(i,j)=dot(data(i,:),data(j,:))/(norm(data(i,:))*norm(data(j,:)));
                
            end
            
            CosD(j,i)=CosD(i,j);
        end
        
   save('CosD.mat')     
    end