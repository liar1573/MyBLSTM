clear
%% change this is enough
%synthpath1 = '/media/zoe/836ec9bd-0427-4076-9c2f-8062ad4c43b7/2017_SCI_1_2/codes/troatBC/results/BLSTM_ratio_Exp_bLSTM_1_hidden_frameLength_23_model_RNN_l2_magnitude_ratio_h512_nFFT256_tanh_dropout0.2_rmsprop_r0.01_batch128_epoch500_val0.1/f001';
synthpath1 = '/home/zoe/Desktop/BC_Baseline/DNN_baseline/test_as_val_129-256-256-129';
synthpath2 = '/home/zoe/Desktop/BC_Baseline/LSTM_baseline/BLSTM_129-129(L)-256-256-129(L)';
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
        
    ac = ac(1:len);
    bc = bc(1:len);
    sy1 = sy1(1:len);
    sy2 = sy2(1:len);
    ac = normalize_wav(ac(1:len));
    bc = normalize_wav(bc(1:len));
    sy1 = normalize_wav(sy1(1:len));
    sy2 = normalize_wav(sy2(1:len));
    denoise_sy1 = specsub(sy1,fs);
    denoise_sy2 = specsub(sy2,fs);
    ac_sy_mcd_dis1 = mcd_distance(ac,sy1,fs);
    ac_bc_mcd_dis = mcd_distance(ac,bc,fs);
    ac_sy_mcd_dis2 = mcd_distance(ac,sy2,fs);
    ac_mcd_dis_denoise_sy1 = mcd_distance(ac,denoise_sy1,fs);
    ac_mcd_dis_denoise_sy2 = mcd_distance(ac,denoise_sy2,fs);


    
    figure;
    subplot(611);spectrogram(ac,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');%set(gca,'ytick',ff)
    subplot(612);title('BC');spectrogram(bc,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(613);spectrogram(sy1,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(614);spectrogram(sy2,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(615);spectrogram(denoise_sy1,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(616);spectrogram(denoise_sy2,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    
    ac_bc_lsd_dis = LogSpectralDistance(ac,bc);
    ac_sy_lsd_dis1 = LogSpectralDistance(ac,sy1);
    ac_sy_lsd_dis2 = LogSpectralDistance(ac,sy2);
    ac_lsd_denoise_sy1 = LogSpectralDistance(ac,denoise_sy1);
    ac_lsd_denoise_sy2= LogSpectralDistance(ac,denoise_sy2);
    
    ac_bc_cep = comp_cep(airfile,htfile);
    ac_sy_cep1 = comp_cep(airfile,synthfile1);
    ac_sy_cep2 = comp_cep(airfile,synthfile2);
%     ac_cep_denoise_sy1 = comp_cep(airfile,synthfile1);
%     ac_cep_denoise_sy2 = comp_cep(airfile,synthfile2);
end