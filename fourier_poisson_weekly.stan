data {
  int<lower=1> N;             // number of rides
  vector[N] t;                // time-of-week in [0, 168)
  int<lower=1> K;             // number of Fourier terms
  real<lower=0> W;            // effective # of weeks observed
}

transformed data {
  real pi2   = 2 * pi();
  real period = 168.0;
}

parameters {
  real alpha0;
  vector[K] alpha;            // cosine coeffs
  vector[K] beta;             // sine coeffs
  real<lower=0> tau;          // shrinkage
}

model {
  // Priors
  alpha0 ~ normal(0, 10);
  tau    ~ inv_gamma(2, 2);   // or your preferred hyperprior

  for (k in 1:K) {
    alpha[k] ~ normal(0, tau / k);
    beta[k]  ~ normal(0, tau / k);
  }

  // Likelihood part 1: log intensity at observed events
  vector[N] log_lambda;
  for (i in 1:N) {
    real f = alpha0;
    for (k in 1:K) {
      f += alpha[k] * cos(pi2 * k * t[i] / period)
         + beta[k]  * sin(pi2 * k * t[i] / period);
    }
    log_lambda[i] = f;
  }
  target += sum(log_lambda);

  // Likelihood part 2: - W * ∫_0^{168} λ(t) dt, via quadrature
  int M = 400;                 // quadrature points
  real dt = period / M;
  real integral = 0;

  for (j in 1:M) {
    real s   = dt * (j - 0.5); // midpoint rule
    real f_s = alpha0;
    for (k in 1:K) {
      f_s += alpha[k] * cos(pi2 * k * s / period)
           + beta[k]  * sin(pi2 * k * s / period);
    }
    integral += exp(f_s);
  }

  target += - W * dt * integral;
}

generated quantities {
  vector[N] log_lik;

  for (i in 1:N) {
    real f = alpha0;
    for (k in 1:K) {
      f += alpha[k] * cos(pi2 * k * t[i] / period)
         + beta[k] * sin(pi2 * k * t[i] / period);
    }
    log_lik[i] = f;
  }
}
