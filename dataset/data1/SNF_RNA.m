function [W]=SKF(Wall,K,t,ALPHA)
K=40;%10-30
t=20;%10-20
ALPHA=0.1;%0.3-0.8
load('JaccL.mat', 'jaccardL');
RNASiNet=load('lncRNA-lncRNA.csv');
load('GSL.mat', 'lncrna_gsSim')
load('CosL.mat', 'CosL')
% ����һ������Ϊ���� cell ����
Wall = cell(1, 3);
matrix1 = jaccardL;
matrix2 = RNASiNet;
matrix3 = lncrna_gsSim;
matrix4 = CosL;
% ��������������� cell �����е�����Ԫ��
Wall{3} = matrix1;
%Wall{2} = matrix2;
Wall{2} =matrix3;
Wall{1} =matrix4;
C = length(Wall);
[m,n]=size(Wall{1});

for i = 1 : C
    newW1{i} = Wall{i}./repmat(sum(Wall{i},1),n,1);
end
sumW1 = zeros(m,n);
for i = 1 : C
    sumW1= sumW1 + newW1{i};
end


for i = 1 : C
    Wall{i} = Wall{i}./repmat(sum(Wall{i},1),n,1);
end

for i = 1 : C
    newW{i} = FindDominateSet(Wall{i},round(K));
end

Wsum = zeros(m,n);
for i = 1 : C
    Wsum = Wsum + Wall{i};
end
for ITER=1:t
    for i = 1 : C           
         Wall{i}=ALPHA*newW{i}*(Wsum - Wall{i})*newW{i}'/(C-1) + (1-ALPHA)*(sumW1 - newW1{i})/(C-1);
    end   
    Wsum = zeros(m,n);
    for i = 1 : C
        Wsum = Wsum + Wall{i};
    end     
end
W = Wsum/C;
w = neighborhood_Com(W,K);
WM= W.*w;
save('RNA_s_kernel-k=40-t=20-ALPHA=0.1.mat','WM')
end

function newW = FindDominateSet(W,K)
[m,n]=size(W);
[YW,IW1] = sort(W,2,'descend');
clear YW;
newW=zeros(m,n);
temp=repmat((1:n)',1,K);
I1=(IW1(:,1:K)-1)*m+temp;
newW(I1(:))=W(I1(:));
newW=newW./repmat(sum(newW,2),1,n);
clear IW1;
clear IW2;
clear temp;
end


function similarities_N = neighborhood_Com(similar_m,kk)
similarities_N=zeros(size(similar_m));

mm = size(similar_m,1);

for ii=1:mm	
	for jj=ii:mm
		iu = similar_m(ii,:);
		iu_list = sort(iu,'descend');
		iu_nearest_list_end = iu_list(kk);
		
		ju = similar_m(:,jj);
		ju_list = sort(ju,'descend');
		ju_nearest_list_end = ju_list(kk);
		if similar_m(ii,jj)>=iu_nearest_list_end & similar_m(ii,jj)>=ju_nearest_list_end
			similarities_N(ii,jj) = 1;
			similarities_N(jj,ii) = 1;
		elseif similar_m(ii,jj)<iu_nearest_list_end & similar_m(ii,jj)<ju_nearest_list_end
			similarities_N(ii,jj) = 0;
			similarities_N(jj,ii) = 0;
		else
			similarities_N(ii,jj) = 0.5;
			similarities_N(jj,ii) = 0.5;
        end
    end
end
end