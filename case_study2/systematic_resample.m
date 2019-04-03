function index=systematic_resample(W)
% Code for systematic resampling

    N=length(W);
    cumW=zeros(N+1,1);
    cumW(2:(N+1))=cumsum(W);
    index=zeros(N,1);
    Numindex=index;
    U=rand(1,1)*(1/N)+(0:(1/N):((N-1)/N));
    
    counter=1;
    for i=1:N
        Numindex(i)=sum(U>cumW(i) & U<=cumW(i+1));
        if Numindex(i)>0
            newcounter=counter+Numindex(i)-1;
            index(counter:newcounter)=i;
            counter=newcounter+1;
        end
    end

end