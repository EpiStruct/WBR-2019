function [model_posterior,CI,l_ev]=Model_Selection_SE2I2R(data,N,tight_prior,parallel,num_cores)
% This function calculates posterior model probabilities for the SEEIIR
% model with three candidate observation models: where data is of daily
% transitions into the exposed class (Post); daily transitions into the
% infectious class (Co); or, daily transitions into the recovered class (Pre). 
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

% The number of exposed and infectious compartments.
shape1=2;
shape2=2;

% Maximum number of samples before code ends.
Max_for_Ev=100000;

% Number of samples from the hidden process per likelohood calculation.
Nx=1000;

% The size of the first batch of samples from the parameter space, before credible
% intervals are calculated.
batch_size=500;

% Quantile used for credible intervals.
significance=0.05;
quant_val=norminv(1-0.5*significance);

% Model parameters specified for calculation using tight-priors
gam=2/3;
sigm=0.5;
bet=0.933;

%% Prior Specification

if tight_prior
    
    % Prior mean and variance for 1/gamma.
    m1=1/gam;
    v1=0.01;
    
    % Prior mean and variance for 1/sigma.
    m2=1/sigm;
    v2=0.01;
    
    % Prior range of R0
    r0_bound=[(bet/gam)-0.003,(bet/gam)+0.003];
    
else
    % Prior mean and variance for 1/gamma.
    m1=2;
    v1=0.75;
    
    % Prior mean and variance for 1/sigma.
    m2=2;
    v2=0.75;
    
    % Prior range of R0
    r0_bound=[1,2];
    
end

% Probability density function of the prior distribution.
prior_pdf=@(r0,sigminv,gaminv) (1/(r0_bound(2)-r0_bound(1)))*(r0<r0_bound(2) & r0>r0_bound(1)).*gampdf(gaminv,(m1^2)/v1,v1/m1).*gampdf(sigminv,(m2^2)/v2,v2/m2);


%% Importance sampling generator and pdf.
% These lines can be changed to allow for other sampling densities if
% necessary. Sampling densities can be model-specific.

IS_rand=@(num_samps) [r0_bound(1)+rand(num_samps,1)*(r0_bound(2)-r0_bound(1)),gamrnd((m2^2)/v2,v2/m2,num_samps,1),gamrnd((m1^2)/v1,v1/m1,num_samps,1)];
IS_pdf=@(r0,sigminv,gaminv) prior_pdf(r0,sigminv,gaminv);

% Proposing samples and computing the sampling
% densities in advanced of likelihood computations.

theta_propPost=IS_rand(Max_for_Ev);
Q_valsPost=IS_pdf(theta_propPost(:,1),theta_propPost(:,2),theta_propPost(:,3));
prior_valsPost=prior_pdf(theta_propPost(:,1),theta_propPost(:,2),theta_propPost(:,3));
l_lh_Post=zeros(Max_for_Ev,1);

theta_propCo=IS_rand(Max_for_Ev);
Q_valsCo=IS_pdf(theta_propCo(:,1),theta_propCo(:,2),theta_propCo(:,3));
prior_valsCo=prior_pdf(theta_propCo(:,1),theta_propCo(:,2),theta_propCo(:,3));
l_lh_Co=zeros(Max_for_Ev,1);

theta_propPre=IS_rand(Max_for_Ev);
Q_valsPre=IS_pdf(theta_propPre(:,1),theta_propPre(:,2),theta_propPre(:,3));
prior_valsPre=prior_pdf(theta_propPre(:,1),theta_propPre(:,2),theta_propPre(:,3));
l_lh_Pre=zeros(Max_for_Ev,1);


%% Pre-specification

% Pre-allocation of variables that keep track of the index of samples and
% the credible intervals
first_counter=ones(3,1);
last_counter=zeros(3,1);
CI=[zeros(3,1),ones(3,1)];

% Pre-specification of normalisation constant.
log_normaliser=NaN;


%% Likelihood Computations
% The algorithm computes sampling weights until either evidence estimates are non-overlapping
% or the cutoff number of iterations is reached.

while max(last_counter)<Max_for_Ev
    
    % Keeping track of the indicies of samples for each model.
    for ii=1:3
        if sum(CI(:,1)>CI(ii,2))<2
            first_counter(ii)=last_counter(ii)+1;
            last_counter(ii)=first_counter(ii)-1+batch_size;
        end
    end
    
    
    %% Pre-allocating vectors for parallel computation
    
    % pre-allocated log-evidence vector
    l_ev=zeros(3,1);
    
    % Log-likelihood vectors for each model.
    l_lh_tempPost=zeros(batch_size,1);
    l_lh_tempCo=zeros(batch_size,1);
    l_lh_tempPre=zeros(batch_size,1);
    
    % Sampling density for each model.
    theta_propPost_temp=theta_propPost(first_counter(1):last_counter(1),:);
    theta_propCo_temp=theta_propCo(first_counter(2):last_counter(2),:);
    theta_propPre_temp=theta_propPre(first_counter(3):last_counter(3),:);
    
    % Prior probability for each model.
    priorPost_temp=prior_valsPost(first_counter(1):last_counter(1),:);
    priorCo_temp=prior_valsCo(first_counter(2):last_counter(2),:);
    priorPre_temp=prior_valsPre(first_counter(3):last_counter(3),:);
    
    
    %% Log-likelihood calculation
    % Log-likelihoods are estimated for each model. These are computed in parallel if specified.
    if parallel
        parpool('local',num_cores)
        parfor ii=1:batch_size
            obs_models=[1,shape1+1,shape1+shape2+1];
            CI_temp2=CI;
            
            %% Post log-likelihood calculation
            % This only continues calculating weights if the CI is overlapping
            % with that of other models.
            if sum(CI_temp2(:,1)>CI_temp2(1,2))<2
                % Checks if the sampled parameter has prior support.
                if priorPost_temp(ii)
                    l_lh_tempPost(ii)=IS_loglikelihood_estimation_SEnInR(theta_propPost_temp(ii,:),shape1,shape2,data,N,obs_models(1),Nx);
                else
                    l_lh_tempPost(ii)=-Inf;
                end
            end
            
            %% Co log-likelihood calculation
            if sum(CI_temp2(:,1)>CI_temp2(2,2))<2
                if priorCo_temp(ii)
                    l_lh_tempCo(ii)=IS_loglikelihood_estimation_SEnInR(theta_propCo_temp(ii,:),shape1,shape2,data,N,obs_models(2),Nx);
                else
                    l_lh_tempCo(ii)=-Inf;
                end
            end
            
            %% Pre log-likelihood calculation
            if sum(CI_temp2(:,1)>CI_temp2(3,2))<2
                if priorPre_temp(ii)
                    l_lh_tempPre(ii)=IS_loglikelihood_estimation_SEnInR(theta_propPre_temp(ii,:),shape1,shape2,data,N,obs_models(3),Nx);
                else
                    l_lh_tempPre(ii)=-Inf;
                end
            end
        end
        delete(gcp('nocreate'))
    else
        obs_models=[1,shape1+1,shape1+shape2+1];
        for ii=1:batch_size
            CI_temp2=CI;
            
            %% Post log-likelihood calculation
            if sum(CI_temp2(:,1)>CI_temp2(1,2))<2
                if priorPost_temp(ii)
                    l_lh_tempPost(ii)=IS_loglikelihood_estimation_SEnInR(theta_propPost_temp(ii,:),shape1,shape2,data,N,obs_models(1),Nx);
                else
                    l_lh_tempPost(ii)=-Inf;
                end
            end
            
            %% Co log-likelihood calculation
            if sum(CI_temp2(:,1)>CI_temp2(2,2))<2
                if priorCo_temp(ii)
                    l_lh_tempCo(ii)=IS_loglikelihood_estimation_SEnInR(theta_propCo_temp(ii,:),shape1,shape2,data,N,obs_models(2),Nx);
                else
                    l_lh_tempCo(ii)=-Inf;
                end
            end
            
            %% Pre log-likelihood calculation
            if sum(CI_temp2(:,1)>CI_temp2(3,2))<2
                if priorPre_temp(ii)
                    l_lh_tempPre(ii)=IS_loglikelihood_estimation_SEnInR(theta_propPre_temp(ii,:),shape1,shape2,data,N,obs_models(3),Nx);
                else
                    l_lh_tempPre(ii)=-Inf;
                end
            end
        end
    end
    
    %% Calculation of evidence, posterior model probabilities and credible intervals
    %  Evidence samples are not computed for a model if its CI is
    %  non-overlapping with other CI's.
    
    % Log-weight and log-evidence computation for Post model.
    if sum(CI(:,1)>CI(1,2))<2
        l_lh_Post(first_counter(1):last_counter(1))=l_lh_tempPost;
        l_IS_weightPost=l_lh_Post(1:last_counter(1))+log(prior_valsPost(1:last_counter(1)))-log(Q_valsPost(1:last_counter(1)));
        l_ev(1)=logsumexp(l_IS_weightPost)-log(last_counter(1));
    end
    
    % Log-weight and log-evidence computation for Co model.
    if sum(CI(:,1)>CI(2,2))<2
        l_lh_Co(first_counter(2):last_counter(2))=l_lh_tempCo;
        l_IS_weightCo=l_lh_Co(1:last_counter(2))+log(prior_valsCo(1:last_counter(2)))-log(Q_valsCo(1:last_counter(2)));
        l_ev(2)=logsumexp(l_IS_weightCo)-log(last_counter(2));
    end
    
    % Log-weight and log-evidence computation for Pre model
    if sum(CI(:,1)>CI(3,2))<2
        l_lh_Pre(first_counter(3):last_counter(3))=l_lh_tempPre;
        l_IS_weightPre=l_lh_Pre(1:last_counter(3))+log(prior_valsPre(1:last_counter(3)))-log(Q_valsPre(1:last_counter(3)));
        l_ev(3)=logsumexp(l_IS_weightPre)-log(last_counter(3));
    end
    
    % Normalises evidence so variance is computable. This constant is computed
    % once so CLT still holds, estimates in future iterations do not strictly sum to 1.
    if isnan(log_normaliser)
        log_normaliser=logsumexp(l_ev);
    end
    
    % Mean and variance of the re-weighted evidence estimates.
    mean_scaled=[mean(exp(l_IS_weightPost-log_normaliser));mean(exp(l_IS_weightCo-log_normaliser));mean(exp(l_IS_weightPre-log_normaliser))];
    var_scaled=[var(exp(l_IS_weightPost-log_normaliser));var(exp(l_IS_weightCo-log_normaliser));var(exp(l_IS_weightPre-log_normaliser))];
    
    % CIs and posterior model probabilities.
    st_dev_est=sqrt(var_scaled./last_counter);
    CI=[mean_scaled-quant_val*st_dev_est,mean_scaled+quant_val*st_dev_est];
    model_posterior=exp(l_ev-logsumexp(l_ev));
    
    
    last_counter
    model_posterior
    CI
    
    % Implementation of the stopping rule: the algorithm stops when
    % credible intervals are non-overlapping.
    max_min=max(CI(:,1));
    uppers=find(CI(:,2)>max_min);
    if length(uppers)==1
        break
    else
        batch_size=500;
    end
end

% Output displayed if estimates did not converge to the specified
% accuracy.
if max(last_counter)>=Max_for_Ev
    display('ERROR: Convergence did not occur quickly enough')
end

