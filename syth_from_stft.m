clear
load('data/pytorch_lstm_one_to_one.mat')
load('data/f001_STFT_TESTSET')
params.win_size_s = 32*8; 
params.fft_length = 256; 
params.hop_size_s = 5*8;
params.fs = 8000;
win = hamming(params.win_size_s);
datapath = 'BC_AC_Dataset\test\ht1';
htname= dir([datapath,'\*.wav']);
    %% denormalize 
for i = 1:numel(result)    
    [datapath,'\',htname(i).name]
    feat = double(result{1,i})';    
    win = hamming(params.win_size_s);
    [data,fs]=audioread([datapath,'\',htname(i).name]);
    [s,p]=compute_features(data,params);
    %sy = overlapaddSPINV(feat,Phase_bc{i},params.fs,win,params);  
    sy = overlapaddSPINV(feat,p,params.fs,win,params);
    des = ['run/hecheng-',htname(i).name];

    audiowrite(des,sy,params.fs)
end