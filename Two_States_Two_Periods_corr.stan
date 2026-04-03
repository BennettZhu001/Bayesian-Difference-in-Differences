data {
  // Panel size (two periods): N = 2 * J
  int<lower=1> N;                         // number of rows (panel observations)
  int<lower=1> J;                         // number of counties
  int<lower=1> K;                         // dim of state predictors (columns in Xbar)

  // Observed outcomes and panel structure
  vector[N] y;                             // outcomes
  array[N] int<lower=1,upper=J> county;    // county index per row
  array[N] int<lower=0,upper=1> post;      // 1 if t=1 else 0
  array[N] int<lower=0,upper=1> D;         // 1 if treated county AND t=1 else 0

  // State mapping & predictors
  array[J] int<lower=1,upper=2> state_of_county; // s[j] in {1 (treated state), 2 (control state)}
  matrix[2, K] Xbar;                               // state-level predictors (rows: s=1,2)

  // Treated indicator per county and an index list for treated counties
  array[J] int<lower=0,upper=1> is_treated;   // 1 if county belongs to treated state, else 0
  int<lower=0> J_treated;                     // number of treated counties
  array[J_treated] int<lower=1,upper=J> treated_idx;  // indices of treated counties

  // Option: center c_j inside the likelihood (subtract mean(c))
  int<lower=0,upper=1> center_c;             // 1 => use (c_j - mean(c)) in mu
}

parameters {
  // Fixed effects
  real A0;
  vector[K] A1;
  real lambda;

  // State random effects (two states: treated s=1, control s=2); non-centered
  vector[2] u_raw;
  real<lower=0, upper=20> sigma_u_treat;   // SD for u_1 (treated state)
  real<lower=0, upper=20> sigma_u_ctrl;    // SD for u_2 (control state)

  // County random effects & treatment effects
  // Treated counties: non-centered bivariate (c_j, delta_j - mu_delta)
  vector[J_treated] zc_treat;              // N(0,1)
  vector[J_treated] zd_treat;              // N(0,1)

  // Control counties: non-centered univariate for c_j
  vector[J - J_treated] zc_ctrl;           // N(0,1)

  // Hyperparameters for treated-pair covariance and mean
  real mu_delta;                                           // mean of delta_j in treated counties
  real<lower=0, upper=20> sigma_c_treat;                   // SD of c_j in treated counties
  real<lower=0, upper=20> sigma_delta;                     // SD of delta_j in treated counties
  real<lower=-1, upper=1> rho;                             // correlation(c_j, delta_j)

  // Control county SD for c_j
  real<lower=0, upper=20> sigma_c_ctrl;

  // Observation noise
  real<lower=0, upper=20> sigma_y;
}

transformed parameters {
  // State random effects
  vector[2] u;
  u[1] = sigma_u_treat * u_raw[1]; // treated state
  u[2] = sigma_u_ctrl  * u_raw[2]; // control state

  // Per-county random effects
  vector[J] c;       // county intercept deviation
  vector[J] delta;   // individual treatment effect (0 for controls)

  // Cholesky factor for treated bivariate (c, delta - mu_delta)
  // L * L' = Sigma, where
  // Sigma = [ sigma_c_treat^2,           rho * sigma_c_treat * sigma_delta
  //           rho * sigma_c_treat * sigma_delta,  sigma_delta^2          ]
  matrix[2,2] L_t;
  {
    real sc = sigma_c_treat;
    real sd = sigma_delta;
    real r  = rho;
    // Numerically safe L for 2x2 covariance
    L_t[1,1] = sc;
    L_t[1,2] = 0;
    L_t[2,1] = r * sd;
    L_t[2,2] = sd * sqrt(1 - r * r + 1e-12); // guard against rounding
  }

  // Fill treated counties
  {
    for (k in 1:J_treated) {
      int j = treated_idx[k];
      // row vector z * L gives row-vector sample
      row_vector[2] z;
      row_vector[2] pair;
      z[1] = zc_treat[k];
      z[2] = zd_treat[k];
      pair = z * L_t;              // ~ N(0, Sigma)
      c[j]     = pair[1];
      delta[j] = mu_delta + pair[2];
    }
  }

  // Fill control counties
  {
    int l = 1;
    for (j in 1:J) {
      if (is_treated[j] == 0) {
        c[j]     = sigma_c_ctrl * zc_ctrl[l];
        delta[j] = 0;
        l += 1;
      }
    }
  }
}

model {
  // Priors for fixed effects
  A0     ~ normal(0, 5);
  A1     ~ normal(0, 2.5);
  lambda ~ normal(0, 5);

  // Non-centered bases
  u_raw    ~ normal(0, 1);
  zc_treat ~ normal(0, 1);
  zd_treat ~ normal(0, 1);
  zc_ctrl  ~ normal(0, 1);

  // Hyperpriors (bounded to [0,20] by declarations; heavy-tailed as in half-Cauchy)
  sigma_u_treat ~ cauchy(0, 20);
  sigma_u_ctrl  ~ cauchy(0, 20);

  sigma_c_treat ~ cauchy(0, 20);
  sigma_c_ctrl  ~ cauchy(0, 20);

  mu_delta      ~ normal(0, 5);
  sigma_delta   ~ cauchy(0, 20);

  // rho has implicit Uniform(-1,1) via bounds and no additional prior

  sigma_y ~ normal(0, 5);

  // Likelihood
  {
    real mean_c = (center_c == 1) ? mean(c) : 0;
    for (n in 1:N) {
      int j = county[n];
      int s = state_of_county[j]; // 1 = treated state, 2 = control state
      real mu_n =
          A0
        + dot_product(row(Xbar, s), A1)
        + u[s]
        + (c[j] - mean_c)
        + lambda * post[n]
        + delta[j] * D[n];
      y[n] ~ normal(mu_n, sigma_y);
    }
  }
}

generated quantities {
  // ATT among treated counties
  real ATT_all;
  {
    real accum = 0;
    for (k in 1:J_treated) accum += delta[treated_idx[k]];
    ATT_all = accum / J_treated;
  }

  // Decomposed state effects U_state = A1' * Xbar[s,] + u[s]
  vector[2] U_state;
  for (s in 1:2) {
    U_state[s] = dot_product(row(Xbar, s), A1) + u[s];
  }

  // For diagnostics (optional): covariance matrix of treated pair
  cov_matrix[2] Sigma_treated;
  Sigma_treated[1,1] = square(sigma_c_treat);
  Sigma_treated[2,2] = square(sigma_delta);
  Sigma_treated[1,2] = rho * sigma_c_treat * sigma_delta;
  Sigma_treated[2,1] = Sigma_treated[1,2];
}
