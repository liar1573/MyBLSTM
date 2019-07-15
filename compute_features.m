function [Spectrum,Phase] = compute_features(dmix,NetParam)
% compute features

 winsize = NetParam.win_size_s; nFFT = NetParam.fft_length; hop = NetParam.hop_size_s;
 % all feature compute logspectrogram for dtw 
        data = spectrogram(dmix,winsize,winsize-hop,nFFT,NetParam.fs,'yaxis');
       % figure;
       %subplot(311); spectrogram(dmix,winsize,winsize-hop,nFFT,NetParam.fs,'yaxis');
        Spectrum= abs(data);
        
        Phase = angle(data); % log(DATA);
end


