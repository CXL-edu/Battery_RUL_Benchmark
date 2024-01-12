clear;clc;
load('Degradation_Prediction_Dataset_ISEA.mat')
sample=[241 472 688 939 1199 1448 1678 1899 2134 2372 2638 2895 3103 3340 3579 3789 3989 4200 4392 4601 4819 5066 5334 5565 5772 5980 6188 6415 6660 6868 7080 7330 7585 7839 8059 8295 8521 8738 8955 9168 9403 9618 9832 10042 10250 10463 10686];
his={TDS.History};
tex={TDS.Target_expanded};
cell=zeros(47,369);
% his=cell2mat(his(1));
for i=1:length(sample)
    ce=cell2mat([his(sample(i)),tex(sample(i))]);
    cell(i,:)=[ce,zeros(1,369-length(ce))];
    plot((1:length(ce)).*5,ce)
    hold on
end
xlabel('cycle')
ylabel('capacity')

fid=fopen('data.csv','a');%创建新的csv文件
if fid<0
	errordlg('File creation failed','Error');
end

fprintf(fid,'%s,',"cycle");
for i=1:length(sample)
    d=strcat('cell_',num2str(i));
    fprintf(fid,'%s,',d);
end
fprintf(fid,'\n');%换行
for j=1:369
    fprintf(fid,'%d,',(j-1)*5);
    for k=1:47
        fprintf(fid,'%d,',cell(k,j));
    end
    fprintf(fid,'\n');%实现没打印一行数据就换行
end
fclose(fid);