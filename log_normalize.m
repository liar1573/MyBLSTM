load('data/f001_STFT_TRAINSET');
% ac=[];
% ac_length=0;
% bc=[];
% bc_length=0;
ac=[];
bc=[];
for i=1:160
    ac=[ac;STFT_ac{1,i}];
    bc=[bc;STFT_bc{1,i}];
%    ac=sum(log(STFT_ac{1,i}),1)+ac;
%    ac_length=size(STFT_ac{1,i},1);
%    bc=sum(log(STFT_bc{1,i}),1)+bc;
%    bc_length=size(STFT_bc{1,i},1);
end
log_STFT_ac_mean=mean(log(ac),1);
log_STFT_bc_mean=mean(log(bc),1);
log_STFT_ac_var=std(log(ac),0,1);
log_STFT_bc_var=std(log(bc),0,1);
STFT_ac_mean=mean(ac,1);
STFT_ac_var=std(ac,0,1);
%save('data/f001_datainfo.mat','log_STFT_ac_mean','log_STFT_bc_mean','log_STFT_ac_var','log_STFT_bc_var');
save('f001_datainfo.mat','log_STFT_ac_mean','log_STFT_bc_mean','log_STFT_ac_var','log_STFT_bc_var','STFT_ac_mean','STFT_ac_var');


