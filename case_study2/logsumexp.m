function log_sum=logsumexp(vec)
% Code for computing the log sum exponential

[max_el,~]=max(vec);
vec=vec-max_el;
log_sum = max_el + log(sum(exp(vec)));

if isinf(max_el)
  log_sum = max_el;
end