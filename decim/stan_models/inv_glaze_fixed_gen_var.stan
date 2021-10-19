data {
    int<lower=0> B; // number of blocks
    int b[B+1]; // integer array with indices of last point locations of block
    int<lower=0> I; // number of decision trials
    int<lower=0> N; // number of point locations
    vector[N] x; // vector with N point locations
    int obs_idx[I];   // integer array with indices of decision point locations
    int obs_decisions[I];
}
parameters {
    real<lower=0, upper=1> H; //Hazardrate used in glaze
    real<lower=0> V; //Variance used in glaze
    //real<lower=0> gen_var; //Variance used in glaze
}
transformed parameters {
    real psi[N];
    real choice_value[N];
    real llr;

    for (i in 1:B) {
        llr = normal_lpdf(x[b[i]+1] | 0.5, 1) -  normal_lpdf(x[b[i]+1] | -0.5, 1);
        psi[b[i]+1] = llr;

        for (n in (b[i]+2):b[i+1]) {
            llr = normal_lpdf(x[n] | 0.5, 1) - normal_lpdf(x[n] | -0.5, 1);
            psi[n] = llr + (psi[n-1] + log( (1 - H) / H + exp(-psi[n-1]))
                    - log((1 - H) / H + exp(psi[n-1])));
            }
    }
}
model {
    H ~ uniform(0,1);
    V ~ normal(0, 50);

    for (i in 1:I) {
        obs_decisions[i] ~ bernoulli(inv_logit(psi[obs_idx[i]]/V));
    }
}

generated quantities {
    real log_lik[I];

    for (i in 1:I) {
        log_lik[i] = bernoulli_logit_lpmf(obs_decisions[i] | psi[obs_idx[i]]/V);
    }
}
