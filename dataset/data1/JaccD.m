function jaccardD = jaccardsim(data)
% 计算向量或矩阵之间的杰卡德相似系数
data=load('disease-lncRNA.csv'); 
rows=size(data,1);
    for i = 1:rows
        for j = 1:rows
            jaccardL(i,j) = length(intersect(data(i,:), data(j,:))) / length(union(data(i,:), data(j,:)));
        end
    end
    save('JaccD.mat')
end
