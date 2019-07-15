params.win_size_s = 32*8; 
params.fft_length = 256; 
params.hop_size_s = 5*8;
params.fs = 8000;
win = hamming(params.win_size_s);
STFT_ac=cell(1,40);
STFT_bc=cell(1,40);
acpath  = 'BC_AC_Dataset\train\air1';
htpath  = 'BC_AC_Dataset\train\ht1';
%acpath  = 'BC_AC_Dataset\test\air1';
%htpath  = 'BC_AC_Dataset\test\ht1';
htname = dir([htpath,'\*.wav']);
airname = dir([acpath,'\*.wav']);
for i=1:numel(airname)
    [acpath,'\',airname(i).name]
    [htpath,'\',htname(i).name]
    [data,fs]=audioread([acpath,'\',airname(i).name]);
    [data_ht,fs_ht]=audioread([htpath,'\',htname(i).name]);
    [s,p]=compute_features(data,params);
    [s_ht,p_ht]=compute_features(data_ht,params);
    s=s';
    s_ht=s_ht';
    STFT_ac{1,i}=s;
    STFT_bc{1,i}=s_ht;
end
save('data/f001_STFT_TRAINSET','STFT_ac','STFT_bc');
%save('data/f001_STFT_TESTSET','STFT_ac','STFT_bc');
