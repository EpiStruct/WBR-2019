% A script to compute the posterior model probabilities of test data sets for the
% Post, Co and Pre models.


% Set 'parallel' to 1 to compute likelihoods in parallel, otherwise set it to 0. 
parallel=0;

% Set 'num_cores' to the number of CPU's to be used if running code in parallel.
num_cores=4;

% Set 'tight_priors' to 1 if computing using tight priors and 0 otherwise. The true parameter values are hardcoded into Model_Selection_SE2I2R  
tight_prior=0;


load('test_set_SIR.mat','data','N')
model_posterior_SIR=Model_Selection_SIkR(data,N,tight_prior,parallel,num_cores);


load('test_set_SI2R.mat','data','N')
model_posterior_SI2R=Model_Selection_SIkR(data,N,tight_prior,parallel,num_cores);


load('test_set_SI5R.mat','data','N')
model_posterior_SI5R=Model_Selection_SIkR(data,N,tight_prior,parallel,num_cores);
