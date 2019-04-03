function [l_weight_mean,Z_saves]=sequential_IS_SEnInR(r0,sigminv,gaminv,shape1,shape2,N,Nx,data,z0,detection_phase)
% The sequential importance resampling for the SE(n)I(n)R model using the 
% importance sampling scheme as per the method of (Black 2018). 
%
%Inputs:
% r0 - Reproduction number.
% sigminv - The mean exposed period.
% gam - The mean infectious period.
% shape1 - The number of exposed compartments in the model.
% shape2 - The number of infection compartments in the model.
% N - The number of individuals in the population.
% Nx - The number of particles.
% data - A vector of observed daily cases of symptom onset after the time of
% the first observation, such that data(1) is the number of cases seen over (0,1].
% z0 - Initial states if particles given in terms of the cumulative number
% of each transition that has occured.
% detection_phase - The index of the transition type that corresponds with
% symptom onset.
%
%Outputs:
% weight_mean: estimate of p(y_t | y_{1:t-1},\theta)
% Z1_saves: state samples of Z1
% Z2_saves: state samples of Z2


% The total number of observations
totalobs=sum(data)+1;

% Specifying the model parameters.
gam=1/gaminv;
sigm=1/sigminv;
bet=r0*gam;

% Setting the current state of particles.
Z_saves=z0;

% Initialising the log likelihood at 0.
l_weight_mean=0;

% Pre-allocating rate vector.
a=NaN(shape1+shape2+1,1);
    
%% Iterations
% Iterating forwards over the time increments and estimating the likelihood
% increments via sequential importance sampling.
for t=1:length(data)
    
    % Calculating the log-weights associated with generating observation times,
    % the mean of the weights will be the estimate of the likelihood
    % increment.
    log_ws=-sum(log(1:data(t)))*ones(1,Nx);
    
    % Pre-allocating arrays for storing forced event times and types.
    tau=NaN(1,max(1,data(t)*(shape1+shape2+1)));
    e_N=tau;
    
    for ii=1:Nx
        
        % Specifying the current state of the particle.
        Z=Z_saves(ii,:);
        
        % Generating observation times over the time increment.
        if data(t)~=0
            tau((data(t)*(shape1+shape2)+1):end)=sort(rand(1,data(t)));
            e_N((data(t)*(shape1+shape2)+1):end)=detection_phase*ones(1,data(t));
        end
        
        % Setting the index of the next forced event.
        stack_indx=data(t)*(shape1+shape2)+1;
        
        % Setting the current rate.
        curr_t=0;
        
        % Noting the next forced event and event time.
        if isnan(tau(end))
            % If there is no forced event the end of the increment is the
            % next forced time. If we are in the last increment there is no
            % upper boundary on time until epidemic fadeout, so we set
            % next_t to infinity.
            if t==length(data)
                next_t=Inf;
                e_n=[];
            else
                next_t=1;
                e_n=[];
            end
        else
            e_n=e_N(stack_indx);
            next_t=tau(stack_indx);
        end
        
        
        % Generate transitions while within the time increment or the
        % epidemic hasn't faded out.
        while (t~=length(data) && curr_t<1) || (t==length(data) && Z(end)~=Z(1))
            
            %%  Calculating original rates of the process.
            a(1)=bet*(N-Z(1)).*(Z(shape1+1)-Z(end))./(N-1);
            a(2:(shape1+1))=shape1*sigm*(Z(1:shape1)-Z(2:(shape1+1)));
            a((shape1+2):end)=shape2*gam*(Z((shape1+1):(shape1+shape2))-Z((shape1+2):end));
            
            %% Generating a forced event
            if ~isempty(e_n)
                forced_rate=a(e_n);
                e_n_new=e_n;
                
                % Checks if either the rate of the next forced
                % event is 0 or if observations correspond to
                % recoveries it checks if observations would lead
                % to epidemic fadeout without the correct number of
                % observations.
                if forced_rate==0 || (e_n_new==(shape1+shape2+1) && (Z(1)-Z(shape1+shape2))==0 && (Z(shape1+shape2)-Z(shape1+shape2+1))==1 && Z(shape1+shape2)~=totalobs)
                    while forced_rate==0 || (e_n_new==(shape1+shape2+1) && (Z(1)-Z(shape1+shape2))==0 && (Z(shape1+shape2)-Z(shape1+shape2+1))==1 && Z(shape1+shape2)~=totalobs)
                        if e_n_new~=1
                            % If the next forced event isn't an
                            % exposure we propose an earlier event
                            % in the chain
                            e_n_new=e_n_new-1;
                        else
                            % If the next forced event is an
                            % exposure we propose an infection
                            e_n_new=shape1+1;
                        end
                        
                        % If the new event leads to an
                        % infeasability we loop and propose another
                        % event (i.e. one earlier in the chain, or
                        % an infection if this proposed transition
                        % is an exposure).
                        forced_rate=a(e_n_new);
                    end
                    
                    % Save the first forced event that gives a
                    % feasible transition.
                    stack_indx=stack_indx-1;
                    e_N(stack_indx)=e_n_new;
                    
                    % Generate the event time and save the weight.
                    tau(stack_indx)=curr_t-(1/forced_rate)*log(1-(1-exp(-forced_rate*(next_t-curr_t)))*rand(1,1));
                    log_ws(ii)=log_ws(ii)-log(forced_rate)+forced_rate*(tau(stack_indx)-curr_t)+log(1-exp(-forced_rate*(next_t-curr_t)));
                    e_n=e_n_new;
                    next_t=tau(stack_indx);
                    
                end
            end
            
            %% Calculating modified rates
            b=a;
            
            % Stop events if they would lead to too many
            % observations prior to die out.
            b(Z==totalobs)=0;
            
            % Set the recovery rate to 0 if it leads to epidemic
            % fade out and the correct number of individuals have not been infected.
            if Z(1)-Z(shape1+shape2+1)<=1 && Z(detection_phase)~=totalobs
                b(shape1+shape2+1)=0;
            end
            
            % Set the rate of the next forced event to 0.
            if ~isempty(e_n)
                b(e_n)=0;
            end
            
            % Set the rate of the observation transition to 0.
            b(detection_phase)=0;
            
            %% Proposing transitions
            
            % Generate transition time.
            sumb=sum(b);
            del_t=-(1/sumb)*log(1-rand(1,1));
            
            if t~=length(data)
                % If we are not at the last increment we are
                % restricting events over (t-1,t].
                if next_t==1 && (del_t+curr_t)>1
                    %% The time increment ends
                    log_ws(ii)=log_ws(ii)-(sum(a)-sumb).*(1-curr_t);
                    curr_t=1;
                elseif del_t<next_t-curr_t
                    %% Generate non-forced event
                    cum_dens=cumsum(b);
                    event=find(rand(1,1)*sumb<cum_dens,1);
                    Z(event)=Z(event)+1;
                    log_ws(ii)=log_ws(ii)+log(a(event)./b(event))-(sum(a)-sumb).*del_t;
                    curr_t=curr_t+del_t;
                elseif del_t>=next_t-curr_t
                    %% Implement the next forced event
                    Z(e_n)=Z(e_n)+1;
                    log_ws(ii)=log_ws(ii)+log(a(e_n))-(sum(a)-sumb).*(next_t-curr_t);
                    curr_t=next_t;
                    e_N(stack_indx)=NaN;
                    tau(stack_indx)=NaN;
                    stack_indx=stack_indx+1;
                    
                    % Update next forced event.
                    if isnan(tau(end))
                        e_n=[];
                        next_t=1;
                    else
                        next_t=tau(stack_indx);
                        e_n=e_N(stack_indx);
                    end
                end
                
            else
                % If the last time increment we allow the time to
                % continue past 1 until the epidemic fades out.
                if del_t<next_t-curr_t
                    %% Generate non-forced event
                    cum_dens=cumsum(b);
                    event=find(rand(1,1)*sumb<cum_dens,1);
                    Z(event)=Z(event)+1;
                    log_ws(ii)=log_ws(ii)+log(a(event)./b(event))-(sum(a)-sumb).*del_t;
                    curr_t=curr_t+del_t;
                elseif del_t>=next_t-curr_t
                    %% Implement the next forced event
                    Z(e_n)=Z(e_n)+1;
                    log_ws(ii)=log_ws(ii)+log(a(e_n))-(sum(a)-sumb).*(next_t-curr_t);
                    curr_t=next_t;
                    e_N(stack_indx)=NaN;
                    tau(stack_indx)=NaN;
                    stack_indx=stack_indx+1;
                    
                    % Update next forced event.
                    if isnan(tau(end))
                        e_n=[];
                        next_t=Inf;
                    else
                        next_t=tau(stack_indx);
                        e_n=e_N(stack_indx);
                    end
                end
            end
        end
        
        Z_saves(ii,:)=Z;
    end
    
    %% Likelihood updates and Resampling
    
    % Calculating the normalised weights.
    l_sumW=logsumexp(log_ws);
    normW=exp(log_ws-l_sumW);
    
    % Systematic resampling of particles.
    resamp=systematic_resample(normW);
    Z_saves=Z_saves(resamp,:);
    
    % Updating the log-likelihood estimate by adding the estimate of the
    % log-likelihood increment.
    l_weight_mean=l_sumW-log(Nx)+l_weight_mean;
    
end