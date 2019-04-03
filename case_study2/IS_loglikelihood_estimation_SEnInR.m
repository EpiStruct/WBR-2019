function l_lh=IS_loglikelihood_estimation_SEnInR(theta,shape1,shape2,data,N,detection_phase,Nx)
% A function which takes a sample from the parameter space,data and a specified
% model in order to run an importance sampling algorithm to estimate the log-likelihood
% of individual outbreaks. These log-likelihoods are combined to give an
% estimate of the overall log-likelihood.

%% Inputs
% theta - The parameter set.
% shape1 - The number of exposed compartments in the model.
% shape2 - The number of infection compartments in the model.
% data - A data set made of a row of cells of the same length as N. Each
% cell represents a data set from an outbreak within a household. Cells contain
% contain a vector of integers that represent the daily number of secondary
% cases of symptom onset within the household.
% N - A vector of household sizes.
% detection_phase - The index of the transition we are observing. For
% example, for the SE(2)I(2)R model, 1 corresponds to the Post model, 3
% corresponds to the Co model and 5 corresponds to the Pre model.
% Nx - The number of simulations of the hidden process used to calculate.
% the likelihood for each outbreak.

%% Outputs
% l_lh - the log-likelihood estimate

%% Log-likeihood calculations 

% Initialising the likelihood function as 1.
l_lh=0;

% For each household compute the likelihood of the outbreak and combine
% to estimate the overall likelihood.
for jj=1:length(N)
    
    % If there are no secondary cases we can comput the likelihood of the outbreak exactly.
    if sum(data{jj})==0 && detection_phase>shape1+1
        initial_l_lh=shape2*log((shape2/(shape2+theta(1))));
        l_lh_hh=0;
    else
        
        % If the observation occurs after the infectious transition then we
        % need to generate initial conditions and compute weights. States are
        % given in terms of the cumulative number of each kind of transition,
        % including a count for the first case.
        if detection_phase>shape1+1
            [Z0,initial_l_lh]=SEnInR_initialstate(detection_phase,theta(1).*(1./theta(3)),1./theta(2),1./theta(3),shape1,shape2,N(jj),Nx,sum(data{jj})+1);
        else
            initial_l_lh=0;
            Z0=[ones(Nx,detection_phase),zeros(Nx,shape1+shape2+1-detection_phase)];
        end
        
        % Importance sampling estimate of the log-likelihood for the outbreak within the household.
        [l_lh_hh,~]=sequential_IS_SEnInR(theta(1),theta(2),theta(3),shape1,shape2,N(jj),Nx,data{jj},Z0,detection_phase);
    end
    
    % Combining the log-likelihoods of households to give the overall
    % log-likelihood.
    l_lh=l_lh+l_lh_hh+initial_l_lh;
    
end
