clear
%% change this is enough
%synthpath = 'run/11';
synthpath = 'run';
%%
bcpath = 'BC_AC_Dataset/test/ht1';
acpath  = 'BC_AC_Dataset/test/air1';
%addpath('/home/zoe/Desktop/BC_Baseline/extra_info/obj_evaluation/obj_evaluation')
%addpath('/home/zoe/Desktop/BC_Baseline/extra_info/mgcfeature')
%addpath('/home/zoe/Desktop/BC_Baseline/extra_info/composite')

synth = dir([synthpath,'/*.wav']);
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
    synthfile = [synthpath,'/',synth(i).name];
    [ac,fs] = audioread(airfile);
    [bc,fs] = audioread(htfile);
    [sy,fs] =  audioread(synthfile);
    if length(ac)>length(sy)
        len = length(sy);
    else
        len = length(ac);
    end
        
    ac = ac(1:len);
    bc = bc(1:len);
    sy = sy(1:len);
    %sy = sy(1:size(ac));
%     ac_bc_lpc_dis = lpc_distance(ac,bc,fs);
%     ac_sy_lpc_dis = lpc_distance(ac,sy,fs);
% ac_sy_mcd_dis = mcd_distance(ac,sy,fs);
%     ac_bc_mcd_dis = mcd_distance(ac,bc,fs);
    

    
    figure;
    subplot(311);spectrogram(ac,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');%set(gca,'ytick',ff)
    subplot(312);title('BC');spectrogram(bc,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    subplot(313);spectrogram(sy,params.win_size_s,params.win_size_s-params.hop_size_s,params.fft_length,fs,'yaxis');
    
   
    
%     ac_bc_lsd_dis = LogSpectralDistance(ac,bc);
%     ac_sy_lsd_dis = LogSpectralDistance(ac,sy);
%     ac_bc_cep = comp_cep(airfile,htfile);
%     ac_sy_cep = comp_cep(airfile,synthfile);
end