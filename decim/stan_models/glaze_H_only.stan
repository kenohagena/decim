data {
  int<lower=0> I; // number of decision trials
  int<lower=0> N; // number of point locations
  int<lower=0, upper=1> y[I]; // subjects answer (0 or 1)
  vector[N] x; // vector with N point locations
  int D[I]; // integer array with indices of decision point locations
}
parameters {
  real<lower=0, upper=1> H; //Hazardrate used in glaze
}
transformed parameters {
  real psi[N];
  real llr;
  llr = normal_lpdf(x[1] | 0.5, 1) - normal_lpdf(x[1] | -0.5, 1);
  psi[1] = llr;

  for (n in 2:N) {

      llr = normal_lpdf(x[n] | 0.5, 1) - normal_lpdf(x[n] | -0.5, 1);
      psi[n] = psi[n-1] + log((1 - H) / H + exp(-psi[n-1]))
             - log((1 - H) / H + exp(psi[n-1]));
      psi[n] = psi[n] + llr;

      }
   }

model {
  H ~ uniform(0,1); //prior on H from truncated normal
  for (i in 1:I) {
    y[i] ~ bernoulli_logit((psi[D[i]]));
 }
}
