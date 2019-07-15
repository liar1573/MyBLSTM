function a = normalize_wav(ac)
    if max(ac)>=abs(min(ac))
       ac = ac/max(ac);
    else
       ac = ac/abs(min(ac)); 
    end
    
    a = ac;