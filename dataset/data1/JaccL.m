function jaccardL = jaccardsim(data)
% �������������֮��Ľܿ�������ϵ��
data=load('disease-lncRNA.csv'); 
data=data.';
rows=size(data,1);
    for i = 1:rows
        for j = 1:rows
            jaccardL(i,j) = length(intersect(data(i,:), data(j,:))) / length(union(data(i,:), data(j,:)));
        end
    end
    save('JaccL')
end

