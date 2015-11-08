data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
  vector[N] alpha;
  vector<lower=0.01>[2] sigma;
  vector<lower=0.01>[2] tau;
}
model {
    for(i in 1:2){
        tau[i] ~ normal(0,sigma[i]);
    }
	for(i in 1:N){
	    y[i] ~ normal(alpha[i],tau[1]);
	}
	for(i in 3:N){
	    alpha[i] ~ normal(2 * alpha[i-1] - alpha[i -2],tau[2]);
	}
	alpha[1] ~ normal(0,100);
	alpha[2] ~ normal(0,100);
	for(i in 1:2){
	    sigma[i] ~ normal(0, 100);
	}
}
