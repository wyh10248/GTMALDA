function [ CosL] = cosSim( data )
%COSSIM Summary of this function goes here
%   Detailed explanation goes here
 data=load('LD_adjmat.txt');  
rows=size(data,1);

    for i=1:rows
        
        for j=1:rows
            
            if (norm(data(i,:))*norm(data(j,:))==0)
                
                CosL(i,j)=0;
                
            else
                CosL(i,j)=dot(data(i,:),data(j,:))/(norm(data(i,:))*norm(data(j,:)));
                
            end
            
            %CosL(j,i)=CosL(i,j);
        end
        
   save('CosL')     
    end