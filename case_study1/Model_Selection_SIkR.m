function [model_posterior,CI,l_ev]=Model_Selection_SIkR(data,N,tight_prior,parallel,num_cores)
% This function calculates posterior model probabilities for the SI(k)R
% model with three candidate observation models, where k=1,2 or 5. 
% This assumes that the prior over candidate models is uniform. Posterior
% model probabilities are calculated such that credible intervals are
% non-overlapping.

%% Inputs
% data - A data set made of a row of cells of the same length as N. Each
% cell represents a data set from an outbreak within a household. Cells contain
% contain a vector of integers that represent the daily number of secondary
% cases of symptom onset within the household.
% N - A vector of household sizes.
% tight_prior - A logical input where 1 runs the algorithm with tight
% priors and 0 runs the algorithm with loose priors. Note: for tight_priors=1 the
% model parameters are hardcoded into the function. 
% parallel - A logical input where 1 runs the algrithm in parallel and 0
% does not.
% num_cores - The number of cores.

%% Outputs
% model_posterior - the posterior model probability estimates. 
% CI - the credible interval estimates up to proportionality for evidence
% estimates.
% l_ev - estimates of the log of evidence.

%% Parameters

% Maximum number of samples before code ends.
Max_for_Ev=100000;

% Number of samples from the hidden process per likelohood calculation.
Nx=500;

% The size of the first batch of samples from the parameter space, before credible
% intervals are calculated.
batch_size=500;

% Quantile used for credible intervals.
significance=0.05;
quant_val=norminv(1-0.5*significance);

% Model parameters specified for calculation using tight-priors
bet=0.933;
gam=2/3;

%% Prior Specification
if tight_prior
    m1=1/gam;
    v1=0.01;
    r0_bound=[bet/gam-0.03,bet/gam+0.03];
else
    m1=2;
    v1=0.75;
    r0_bound=[1,2];
end

prior_pdf=@(r0,gaminv) (1/(r0_bound(2)-r0_bound(1)))*(r0<r0_bound(2) & r0>r0_bound(1)).*gampdf(gaminv,(m1^2)/v1,v1/m1);
prior_rand=@(num_samps) [r0_bound(1)+rand(num_samps,1)*(r0_bound(2)-r0_bound(1)),gamrnd((m1^2)/v1,v1/m1,num_samps,1)];

%% Sampling with IS density and prior calculations
% The IS density need not be the same as the prior and may be model
% specific.

theta_prop1=prior_rand(Max_for_Ev);
prior_vals1=prior_pdf(theta_prop1(:,1),theta_prop1(:,2));
Q_vals1=prior_pdf(theta_prop1(:,1),theta_prop1(:,2));
l_lh_given_theta1=zeros(Max_for_Ev,1);

theta_prop2=prior_rand(Max_for_Ev);
prior_vals2=prior_pdf(theta_prop2(:,1),theta_prop2(:,2));
Q_vals2=prior_pdf(theta_prop2(:,1),theta_prop2(:,2));
l_lh_given_theta2=zeros(Max_for_Ev,1);

theta_prop5=prior_rand(Max_for_Ev);
prior_vals5=prior_pdf(theta_prop5(:,1),theta_prop5(:,2));
Q_vals5=prior_pdf(theta_prop5(:,1),theta_prop5(:,2));
l_lh_given_theta5=zeros(Max_for_Ev,1);

%% Initialisation for computations of likelihoods and evidence

% Keeping track of the number of iterations
counter=zeros(1,3);

% Pre-allocation of likelihood vectors
l_lh_given_theta_temp1=zeros(batch_size,1);
l_lh_given_theta_temp2=zeros(batch_size,1);
l_lh_given_theta_temp5=zeros(batch_size,1);

% Pre-allocation for credible intervals and log-evidence.
CI=[zeros(3,1),ones(3,1)];
l_ev=zeros(3,1);

%% Estimating Evidence
% This calculates importance weights until evidence estimates are
% non-overlapping or the maximum number of samples is exceeded.
while (max(counter)*batch_size)<Max_for_Ev
    
    % Keeping track of the index of the number of samples for each model.
    for ii=1:3
        if sum(CI(:,1)>CI(ii,2))<2
            counter(ii)=counter(ii)+1;
        end
    end
    
    % Pre-allocation of vectors to avoid overhead in parallel computation.
    theta_prop1_temp=theta_prop1(1:batch_size+batch_size*(counter(1)-1),:);
    theta_prop2_temp=theta_prop2(1:batch_size+batch_size*(counter(2)-1),:);
    theta_prop5_temp=theta_prop5(1:batch_size+batch_size*(counter(3)-1),:);
    
    prior_vals1_temp=prior_vals1(1:batch_size+batch_size*(counter(1)-1));
    prior_vals2_temp=prior_vals2(1:batch_size+batch_size*(counter(2)-1));
    prior_vals5_temp=prior_vals5(1:batch_size+batch_size*(counter(3)-1));
    
    %% Estimating the log-likelihoods
    % Log-likelihoods are estimated for each model. These are computed in parallel if specified.
    if parallel
        parpool('local',num_cores)
        parfor ii=1:batch_size
            % Setting temporary variables to avoid overhead from broadcast variables.
            data_temp=data;
            CI_temp=CI;
            
            % SIR calculation of the log-likelihood if the credible interval is
            % still overlapping with others and if the sample has prior
            % support.
            if sum(CI_temp(:,1)>CI_temp(1,2))<2
                if prior_vals1_temp(ii)
                    % Looping over each household and combining the
                    % individual outbreak likelihoods.
                    for jj=1:length(N)
                        l_lh_hh=sequential_IS_SIkR(theta_prop1_temp(ii,:),N(jj),Nx,data_temp{jj},1);
                        l_lh_given_theta_temp1(ii)=l_lh_given_theta_temp1(ii)+l_lh_hh ;
                    end
                else
                    l_lh_given_theta_temp1(ii)=-Inf
                end
            end
            
            % SI(2)R calculation of the log-likelihood.
            if sum(CI_temp(:,1)>CI_temp(2,2))<2
                if prior_vals2_temp(ii)
                    for jj=1:length(N)
                        l_lh_hh=sequential_IS_SIkR(theta_prop2_temp(ii,:),N(jj),Nx,data_temp{jj},2);
                        l_lh_given_theta_temp2(ii)=l_lh_given_theta_temp2(ii)+l_lh_hh ;
                    end
                else
                    l_lh_given_theta_temp2(ii)=-Inf
                end
            end
            
            % SI(5)R calculation of the log-likelihood.
            if sum(CI_temp(:,1)>CI_temp(3,2))<2
                if prior_vals5_temp(ii)
                    for jj=1:length(N)
                        l_lh_hh=sequential_IS_SIkR(theta_prop5_temp(ii,:),N(jj),Nx,data_temp{jj},5);
                        l_lh_given_theta_temp5(ii)=l_lh_given_theta_temp5(ii)+l_lh_hh ;
                    end
                else
                    l_lh_given_theta_temp5(ii)=-Inf
                end
            end
        end
        delete(gcp('nocreate'))
    else
        for ii=1:batch_size
            
            % SIR calculation of the log-likelihood.
            if sum(CI(:,1)>CI(1,2))<2
                if prior_vals1_temp(ii)
                    for jj=1:length(N)
                        l_lh_hh=sequential_IS_SIkR(theta_prop1_temp(ii,:),N(jj),Nx,data{jj},1);
                        l_lh_given_theta_temp1(ii)=l_lh_given_theta_temp1(ii)+l_lh_hh ;
                    end
                else
                    l_lh_given_theta_temp1(ii)=-Inf;
                end
            end
            
            % SI(2)R calculation of the log-likelihood.
            if sum(CI(:,1)>CI(2,2))<2
                if prior_vals2_temp(ii)
                    for jj=1:length(N)
                        l_lh_hh=sequential_IS_SIkR(theta_prop2_temp(ii,:),N(jj),Nx,data{jj},2);
                        l_lh_given_theta_temp2(ii)=l_lh_given_theta_temp2(ii)+l_lh_hh ;
                    end
                else
                    l_lh_given_theta_temp2(ii)=-Inf;
                end
            end
            
            % SI(5)R calculation of the log-likelihood.
            if sum(CI(:,1)>CI(3,2))<2
                if prior_vals5_temp(ii)
                    for jj=1:length(N)
                        l_lh_hh=sequential_IS_SIkR(theta_prop5_temp(ii,:),N(jj),Nx,data{jj},5);
                        l_lh_given_theta_temp5(ii)=l_lh_given_theta_temp5(ii)+l_lh_hh ;
                    end
                else
                    l_lh_given_theta_temp5(ii)=-Inf;
                end
            end
        end
    end
    
    %% Computing the log evidence estimate for the SIR model.
    if sum(CI(:,1)>CI(1,2))<2
        l_lh_given_theta1((1:batch_size)+batch_size*(counter(1)-1))=l_lh_given_theta_temp1;
        l_IS_weight1=l_lh_given_theta1(1:(batch_size*counter(1)))+log(prior_vals1(1:(batch_size*counter(1))))-log(Q_vals1(1:(batch_size*counter(1))));
        l_ev(1)=logsumexp(l_IS_weight1)-log(batch_size*counter(1));
    end
    
    
    %% Computing the log evidence estimate for the SI(2)R model.
    if sum(CI(:,1)>CI(2,2))<2
        l_lh_given_theta2((1:batch_size)+batch_size*(counter(2)-1))=l_lh_given_theta_temp2;
        l_IS_weight2=l_lh_given_theta2(1:(batch_size*counter(2)))+log(prior_vals2(1:(batch_size*counter(2))))-log(Q_vals2(1:(batch_size*counter(2))));
        l_ev(2)=logsumexp(l_IS_weight2)-log(batch_size*counter(2));
    end
    
    %% Computing the log evidence estimate for the SI(5)R model.
    if sum(CI(:,1)>CI(3,2))<2
        l_lh_given_theta5((1:batch_size)+batch_size*(counter(3)-1))=l_lh_given_theta_temp5;
        l_IS_weight5=l_lh_given_theta5(1:(batch_size*counter(3)))+log(prior_vals5(1:(batch_size*counter(3))))-log(Q_vals5(1:(batch_size*counter(3))));
        l_ev(3)=logsumexp(l_IS_weight5)-log(batch_size*counter(3));
    end
        log_normaliser=logsumexp(l_ev);
    
    %% Calculating CI's and posterior model probabilities
    scaled_weight1=exp(l_IS_weight1-log_normaliser);
    scaled_weight2=exp(l_IS_weight2-log_normaliser);
    scaled_weight5=exp(l_IS_weight5-log_normaliser);
    
    mean_scaled=[mean(scaled_weight1);mean(scaled_weight2);mean(scaled_weight5)];
    
    var_1=var(scaled_weight1);
    var_2=var(scaled_weight2);
    var_5=var(scaled_weight5);
    
    
    st_dev_est=[sqrt(var_1/(batch_size*counter(1)));sqrt(var_2/(batch_size*counter(2)));sqrt(var_5/(batch_size*counter(3)))];
    CI=[mean_scaled-quant_val*st_dev_est,mean_scaled+quant_val*st_dev_est];
    
    model_posterior=exp(l_ev-logsumexp(l_ev));
    
    %% Check if stopping condition holds
    max_min=max(CI(:,1));
    uppers=find(CI(:,2)>max_min);
    if length(uppers)==1
        break
    end
    
    counter
    model_posterior
    CI

end

%% Error message if convergence did not occur
if (max(counter)*batch_size)>=Max_for_Ev
    display('ERROR: Convergence did not occur quickly enough')
end
