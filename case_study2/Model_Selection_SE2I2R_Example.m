% A script to compute the posterior model probabilities of test data sets for the
% Post, Co and Pre models.


% Set 'parallel' to 1 to compute likelihoods in parallel, otherwise set it to 0. 
parallel=1;

% Set 'num_cores' to the number of CPU's to be used if running code in parallel.
num_cores=4;

% Set 'tight_priors' to 1 if computing using tight priors and 0 otherwise. The true parameter values are hardcoded into Model_Selection_SE2I2R  
tight_prior=0;


% load('test_set_SEEIIR_Post.mat','data','N')
% model_posterior_Post=Model_Selection_SE2I2R(data,N,tight_prior,parallel,num_cores);
% 
% 
% load('test_set_SEEIIR_Co.mat','data','N')
% model_posterior_Co=Model_Selection_SE2I2R(data,N,tight_prior,parallel,num_cores);
% 

load('test_set_SEEIIR_Pre.mat','data','N')
model_posterior_Pre=Model_Selection_SE2I2R(data,N,tight_prior,parallel,num_cores);


