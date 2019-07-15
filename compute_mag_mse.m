clear
%% change this is enough
synthpath1 = '/home/zoe/Desktop/BC_Baseline/LSTM_baseline/try';
synthpath2 = '/home/zoe/Desktop/BC_Baseline/DNN_baseline/129-256-256-129';
%%
bcpath = '/home/zoe/Desktop/BC_Baseline/extra_info/8kf001/test/ht1';
acpath  = '/home/zoe/Desktop/BC_Baseline/extra_info/8kf001/test/air1';
addpath('/home/zoe/Desktop/BC_Baseline/extra_info/obj_evaluation/obj_evaluation')
addpath('/home/zoe/Desktop/BC_Baseline/extra_info/mgcfeature')
addpath('/home/zoe/Desktop/BC_Baseline/extra_info/composite')

synth = dir([synthpath1,'/*.wav']);
htname = dir([bcpath,'/*.wav']);
airname = dir([acpath,'/*.wav']);
params.win_size_s = 32*8; 
params.fft_length = 256; 
params.hop_size_s = 5*8;
params.fs = 8000;
win = hamming(params.win_size_s);
for i = 1:numel(airname)
    airfile = [acpath,'/',airname(i).name];
    htfile = [bcpath,'/',htname(i).name];
    synthfile1 = [synthpath1,'/',synth(i).name];
    synthfile2 = [synthpath2,'/',synth(i).name];
    [ac,fs] = audioread(airfile);
    [bc,fs] = audioread(htfile);
    [sy1,fs] =  audioread(synthfile1);
    [sy2,fs] =  audioread(synthfile2);
    if length(ac)>length(sy1)
        len = length(sy1);
    else
        len = length(ac);
    end
        
    ac = normalize_wav(ac(1:len));
    bc = normalize_wav(bc(1:len));
    sy1 = normalize_wav(sy1(1:len));
    sy2 = normalize_wav(sy2(1:len));
    
    acc = 20*log10(compute_features(ac,params));
    sy1_c = 20*log10(compute_features(sy1,params));
    sy2_c = 20*log10(compute_features(sy2,params));
    mse_sy1 = sum((acc-sy1_c).^2,2)/size(sy1_c,2);
    mse_sy1_low = sum(mse_sy1(1:64))/64;
    mse_sy1_high = sum(mse_sy1(65:end))/65;
    mse_sy2 = sum((acc-sy2_c).^2,2)/size(sy2_c,2);
    mse_sy2_low = sum(mse_sy2(1:64))/64;
    mse_sy2_high = sum(mse_sy2(65:end))/65;
    close all
    figure;
    select = 10;
    plot(acc(select,:))
    hold on
    plot(sy1_c(select,:),'r')
    
    figure;
    select = 119;
    plot(acc(select,:))
    hold on
    plot(sy1_c(select,:),'r')
%     hold on
%     plot(sy2_c(select,:),'k')
%     hold on
%     plot(sy2_c(:,select))
    
    
    
    figure;
    subplot(411);spectrogram(ac,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');%set(gca,'ytick',ff)
    subplot(412);title('BC');spectrogram(bc,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(413);spectrogram(sy1,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(414);spectrogram(sy2,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');

end