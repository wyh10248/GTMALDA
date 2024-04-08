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
    % ��ʼ��circRNA��disease��������
%     ���д�����MATLAB�е���˼������һ����СΪrows x k���������C��
%     �����е�Ԫ���Ǵ�[0,1)��Χ�ھ��ȷֲ����������
%     Ȼ��Ծ����е�ÿ��Ԫ��ȡ����ֵ��abs��������
%     ���仰˵���ô���������һ���Ǹ��������������ÿ��Ԫ�ض���0��1֮����������
    C = abs(rand(rows,k));
    D = abs(rand(cols,k));
    %[test_results(fold), ~] = improve_NMFLP(X_test,circSimi,disSimi,0.01,0.7,300);
    % ����ԽǾ���
    diag_cf=diag(sum(lncSimi,2));
    diag_df=diag(sum(disSimi,2));
     % ������������ݶԾ���ֽ��Ӱ�죬��˶��ڽӾ�������ع�
    A1=lncSimi*A;
    A2=A*disSimi;
    for i=1:rows
        for j=1:cols
            A(i,j)=max(A1(i,j),A2(i,j));
        end
    end
    
    % ��һ��
    AA=zeros(rows,cols);
    for j =1:cols
        colList=A(:,j);
        for i =1:rows
        AA(i,j)=(A(i,j)-min(colList))/(max(colList)-min(colList));
        end
    end
    A=AA;
    
    % ³���Ծ���ֽⲿ��
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