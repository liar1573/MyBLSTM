function syn=overlapaddSPINV(X,ph,fs,win,params)

%spectrogram inversion and overlap add

syn=zeros((size(X,2)-1)*(params.hop_size_s)+params.win_size_s,1);
overlapsig=zeros(size(syn));


for t=1:size(X,2)
  
  b=X(:,t).*exp(i*ph(:,t));

  if mod(params.fft_length,2)
    b=[b; conj(b(end:-1:2))];
  else
    b=[b; conj(b(end-1:-1:2))];
  end
  
  frame=real(ifft(b));

  indext=(t-1)*params.hop_size_s+1:(t-1)*params.hop_size_s+length(win);
  
  syn(indext)=syn(indext)+frame;
  overlapsig(indext)=overlapsig(indext)+win;
  
end

%compensate windowing
syn=syn./overlapsig;

