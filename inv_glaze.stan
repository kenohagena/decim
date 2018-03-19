data {    
    int<lower=0> I; // number of decision trials
    int<lower=0> N; // number of point locations
    vector[N] x; // vector with N point locations
    int obs_idx[I];   // integer array with indices of decision point locations
    int obs_decisions[I];
    
    }

parameters {
    real<lower=0, upper=1> H; //Hazardrate used in glaze
    real<lower=1> V; //Variance used in glaze
    real<lower=0> gen_var; //Variance used in glaze
    
}

transformed parameters{
    
    real psi[N];
    real choice_value[N];
    real llr;

    
    llr = normal_lpdf(x[1] | 0.5, gen_var) -  normal_lpdf(x[1] | -0.5, gen_var);
    psi[1] = llr;
    for (i in 2:N) {
            llr = normal_lpdf(x[i] | 0.5, gen_var) - normal_lpdf(x[i] | -0.5, gen_var);
            psi[i] = llr + (psi[i-1] + log( (1 - H) / H + exp(-psi[i-1]))
                    - log((1 - H) / H + exp(psi[i-1])));
    }
    /*
    llr = normal_lpdf(x[1] | 0.5, 1) -  normal_lpdf(x[1] | -0.5, 1);
    psi[1] = llr;
    for (i in 2:N) {
            llr = normal_lpdf(x[i] | 0.5, 1) - normal_lpdf(x[i] | -0.5, 1);
            psi[i] = llr + (psi[i-1] + log( (1 - H) / H + exp(-psi[i-1]))
                    - log((1 - H) / H + exp(psi[i-1])));
    }
    */
}

model {
    H ~ beta(1,3); // normal(0, 20)
    V ~ normal(1, 5);
    gen_var ~ normal(1, 10);

    for (i in 1:I) {
        obs_decisions[i] ~ bernoulli(inv_logit(psi[obs_idx[i]]/V));
    }
}