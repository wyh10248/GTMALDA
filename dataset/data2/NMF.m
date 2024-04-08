function [P] = NMF(A,lncSimi,disSimi,beita,k,iterate)
    %ALP=A;
    A=load('LD_adjmat.txt');
    load('RNA_s_kernel-k=10-t=10.mat', 'WM');
    lncSimi = WM;
    load('diease_similarity_kernel-k=10-t=10.mat', 'WM');
    disSimi = WM;
    beita = 0.01;
    k = 64;
    iterate = 500;
    [rows,cols]=size(A);%[585*88]
    % 初始化circRNA和disease迭代矩阵
%     这行代码在MATLAB中的意思是生成一个大小为rows x k的随机矩阵C，
%     矩阵中的元素是从[0,1)范围内均匀分布的随机数，
%     然后对矩阵中的每个元素取绝对值（abs函数）。
%     换句话说，该代码生成了一个非负的随机矩阵，其中每个元素都是0到1之间的随机数。
    C = abs(rand(rows,k));
    D = abs(rand(cols,k));
    %[test_results(fold), ~] = improve_NMFLP(X_test,circSimi,disSimi,0.01,0.7,300);
    % 计算对角矩阵
    diag_cf=diag(sum(lncSimi,2));
    diag_df=diag(sum(disSimi,2));
     % 避免假阴性数据对矩阵分解的影响，因此对邻接矩阵进行重构
    A1=lncSimi*A;
    A2=A*disSimi;
    for i=1:rows
        for j=1:cols
            A(i,j)=max(A1(i,j),A2(i,j));
        end
    end
    
    % 归一化
    AA=zeros(rows,cols);
    for j =1:cols
        colList=A(:,j);
        for i =1:rows
        AA(i,j)=(A(i,j)-min(colList))/(max(colList)-min(colList));
        end
    end
    A=AA;
    
    % 鲁棒性矩阵分解部分
    for step = 1:iterate
        Y=A-C*D';
        B=L21_norm(Y);
        
        % for circRNA coding
        if  beita >0
            BAD=B*A*D+beita*lncSimi*C;
            BCDD=B*C*(D')*D+beita*diag_cf*C;
        end
        C=C.*(BAD./BCDD);
        
        %for disease coding
        if  beita >0
            ABC=(A')*B*C+beita*disSimi*D;
            DCBC=D*(C')*B*C+beita*diag_df*D;
        end
        D=D.*(ABC./DCBC);
        
        scoreMat_NMF=A-C*D';
        error=mean(mean(abs(scoreMat_NMF)))/mean(mean(A));
        fprintf('step=%d  error=%f\n',step,error);
    end
     save C C;
     save D D;