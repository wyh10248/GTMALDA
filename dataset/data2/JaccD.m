function jaccardD = jaccardsim(data)
% �������������֮��Ľܿ�������ϵ��
data=load('LD_adjmat.txt');  
data=data.'
rows=size(data,1);
    for i = 1:rows
        for j = 1:rows
            jaccardD(i,j) = length(intersect(data(i,:), data(j,:))) / length(union(data(i,:), data(j,:)));
        end
    end
    save('JaccD')
end
