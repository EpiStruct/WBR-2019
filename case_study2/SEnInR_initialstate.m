function [Z,l_mean_weights]=SEnInR_initialstate(detection_phase,bet,sigm,gam,shape1,shape2,N,Nx,totalobs)
% A function which takes an observation model and total number of infected
% individuals and returns appropriate initial states and the likelihood associated
% with observing an appropriate initial condition.

%% Inputs
% detection_phase - The index of the transition we are observing. For
% example, for the SE(2)I(2)R model, 1 corresponds to the Post model, 3
% corresponds to the Co model and 5 corresponds to the Pre model.\
% bet - Beta model parameter.
% sigm - Sigma model parameter.
% gam - Gamma model parameter.
% N - The size of the population.
% Nx - The number of initial conditions to generate.
% totalobs - The total number of observations (including the initial
% observation).

%% Outputs
% Z - states given in terms of the cumulative number of each kind of
% transition.
% l_mean_weights - The log of the mean IS weights of the particles.

%% Pre-allocating variables
Z=zeros(Nx,shape1+shape2+1,1);
r=zeros(1,shape1+shape2+1);
log_weights=zeros(Nx,1);

%% Generate an initial condition for each particle
for jjjj=1:Nx
    % States given as (E1,E2,I1,I2,R), where S=N-E1-E2-I1-I2-R.
    state=zeros(1,shape1+shape2+1);
    
    % Begin from E1=1.
    state(1)=1;
    
    % If exposures are observed the initial condition is given.
    if detection_phase==1
        Z(jjjj,1)=1;
    else
        
        % While the epidemic has not died out, continue to generate an
        % initial condition.
        while 1
            
            %% Calculating initial rates
            r(1)=bet*(N-sum(state))*sum(state((shape1+1):(shape1+shape2)))/(N-1);
            r(2:(shape1+1))=shape1*sigm*state(1:shape1);
            r((shape1+2):(shape1+shape2+1))=shape2*gam*state((shape1+1):(shape1+shape2));
            
            
            %% Calculating modified rates
            b=r;
            for zzz=1:detection_phase
                if sum(state(zzz:end))==totalobs
                    b(zzz)=0;
                end
            end
            
            if sum(state(1:(end-1)))==1 && totalobs>1
                b(shape1+shape2+1)=0;
            end
            
            %% Generating transition and updating weight
            cumdens=cumsum(b);
            trans=rand(1,1);
            trans_type=find(trans*cumdens(end)<cumdens,1);
            if trans_type~=1
                state(trans_type-1)=state(trans_type-1)-1;
            end
            state(trans_type)=state(trans_type)+1;
            log_weights(jjjj)=log_weights(jjjj)+log(r(trans_type)/sum(r))-log(b(trans_type)/sum(b));
            
            % Ending if the transition is an observation.
            if trans_type==detection_phase
                break
            end
        end
    end
    
    % Representing state in terms of cumulative number of events.
    for ii=1:(shape1+shape2+1)
        Z(jjjj,ii)=sum(state(ii:end));
    end
end

% Calculating normalised weights and resampling.
l_sumW=logsumexp(log_weights);
normW=exp(log_weights-l_sumW);
resamp=systematic_resample(normW);
Z=Z(resamp,:);
l_mean_weights=l_sumW-log(Nx);
