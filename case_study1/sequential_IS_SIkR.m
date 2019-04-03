function [log_lh,Z_saves]=sequential_IS_SIkR(theta,N,Nx,data,shape)
% The sequential importance resampling for the SI(k)R model using the
% importance sampling scheme as per the method of (Black 2018).
%
%Inputs:
% theta - Parameter values of the particle.
% N - Number of individuals in the population.
% Nx - Number of state particles.
% data - The daily number of observed cases of infection within a household
% after the first observation, that is, the first entry corresponds to time
% increment (0,1].
% shape - the number of infectious compartments in the model.
%
%Outputs:
% log_lh - Estimate of the log-likelihood of an individual outbreak.
% Z_saves - States of the particles given in terms of the cumulative number
% of transitions of each kind.

% Giving the parameters in terms of model rates.
gam=1/theta(2);
bet=theta(1)*gam;

% Pre-allocating log-likelihood and the initial states and rate vector.
log_lh=0;
Z_saves=[ones(Nx,1),zeros(Nx,shape)];
a=NaN(shape+1,1);

%% Iterations
% Iterating forwards over the time increments and estimating the likelihood
% increments via sequential importance sampling.
for t=1:length(data)
    
    % Calculating the log-weights associated with generating observation times,
    % the mean of the weights will be the estimate of the likelihood
    % increment.
    log_ws=-sum(log(1:data(t)))*ones(Nx,1);
    

    for ii=1:Nx
        
        % Generating observation times and truncating with the end of the
        % time increment to make simpler.
        tau=sort(rand(1,data(t)));
        
        if t~=length(data)
            tau_n_bound=[tau,1];
        else
            tau_n_bound=[tau,Inf];
        end
        
        % Setting the current time, next observation time and state.
        curr_t=0;
        next_t=tau_n_bound(find(tau_n_bound>=curr_t,1));
        Z=Z_saves(ii,:);
        
        % Keep generating transitions until at the end of the time
        % increment or on the last time increment with epidemic fadeout.
        while (t~=length(data) && curr_t<1) || (t==length(data) && Z(1)~=Z(end))
            
            %% Calculate original rates
            a(1)=bet*(N-Z(1)).*(Z(1)-Z(end))./(N-1);
            a(2:end)=shape*gam*(Z(1:(end-1))-Z(2:end));
            
            %% Calculate modified rates
            b=a;
            b(1)=0;
            if  t==length(data)
                b(end)=a(end)*((Z(1)-Z(end))>1 || Z(1)==(1+sum(data)));
            else
                b(end)=a(end)*((Z(1)-Z(end))>1);
            end
            
            %% Propose transitions
            sumb=sum(b);
            del_t=-(1/sumb)*log(1-rand(1,1));
            
                if next_t==1 && (del_t+curr_t)>1
                    %% Move to the end of the time increment 
                    % This only occurs if this is not the last time increment.
                    log_ws(ii)=log_ws(ii)-(sum(a)-sumb).*(1-curr_t);
                    curr_t=1;
                elseif del_t<next_t-curr_t
                    %% Generate a non-observed event
                    event_index=find(cumsum(b)>=sumb*rand(1,1),1);
                    Z(event_index)=Z(event_index)+1;
                    log_ws(ii)=log_ws(ii)+log(a(event_index)./b(event_index))-(sum(a)-sumb).*del_t;
                    curr_t=curr_t+del_t;
                elseif del_t>=next_t-curr_t
                    %% Implement the next observed event
                    Z(1)=Z(1)+1;
                    log_ws(ii)=log_ws(ii)+log(a(1))-(sum(a)-sumb).*(next_t-curr_t);
                    curr_t=next_t;
                    next_t=tau_n_bound(find(tau_n_bound>curr_t,1));
                end

        end
        Z_saves(ii,:)=Z;
    end
    
    %% Resample particles
    log_sumW=logsumexp(log_ws);
    normW=exp(log_ws-log_sumW);
    resamp=systematic_resample(normW);
    Z_saves=Z_saves(resamp,:);
    
    %% Update log-likelihood estimate
    log_lh=log_sumW-log(Nx)+log_lh;
    
end
