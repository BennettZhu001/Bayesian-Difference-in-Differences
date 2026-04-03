data {
  int<lower=1> N;                               // observations (2 * J)
  int<lower=1> J;                               // counties
  int<lower=1> S;                               // states
  int<lower=1> K;                               // dim of state predictors

  vector[N] y;                                  // outcomes
  array[N] int<lower=1,upper=J> county;         // county index per row
  array[N] int<lower=0,upper=1> post;           // 1 if t=1
  array[N] int<lower=0,upper=1> D;              // 1 if treated county & t=1

  array[J] int<lower=1,upper=S> state_of_county;// s[j] in {1..S}
  matrix[S, K] Xbar;                            // state-level predictors

  array[J] int<lower=0,upper=1> is_treated;     // county in treated state?
  array[S] int<lower=0,upper=1> state_is_treated; // state indicator (0=control,1=treated)

  int<lower=1> J_treated;                       // number of treated counties
  array[J_treated] int<lower=1,upper=J> treated_idx;

  // Options
  int<lower=0,upper=1> center_c;                // 1 => use (c - mean(c)) in mu
  int<lower=0,upper=1> pool_u_within_group;     // 0: no pooling of u_s (very weak prior per state)
                                                // 1: pool u_s within group (treated vs control)
  real<lower=0> big_sd_u;                       // "diffuse prior" SD for u_s when not pooled (e.g., 1e3)
}

parameters {
  // Fixed effects
  real A0;
  vector[K] A1;
  real lambda;

  // ----- State random effects u_s -----
  // Non-centered bases always present
  vector[S] z_u;

  // If pooling within group: estimate group-specific scales
  real<lower=1e-6, upper=10> sigma_u_treat;
  real<lower=1e-6, upper=10> sigma_u_ctrl;

  // If not pooling within group: free u_s parameters (we'll still use z_u to stabilize)
  // (No extra parameters needed; we will scale by data big_sd_u.)

  // ----- County random effects c_j (state-specific pooling) -----
  vector[J] z_c;                 // std normals
  real<lower=1e-6, upper=10> sigma_c_treat;   // SD for c_j in treated states
  real<lower=1e-6, upper=10> sigma_c_ctrl;    // SD for c_j in control states

  // Treatment heterogeneity among treated counties (non-centered)
  vector[J_treated] z_delta;
  real mu_delta;                 // ATT among treated counties
  real<lower=1e-6, upper=10> sigma_delta;
  // Observation noise
  real<lower=1e-6, upper=10> sigma_y;
}

transformed parameters {
  // ----- Build state RE u_s -----
  vector[S] u;
  for (s in 1:S) {
    if (pool_u_within_group == 1) {
      // Pool only within treated / control groups (separate scales)
      real scale = state_is_treated[s] == 1 ? sigma_u_treat : sigma_u_ctrl;
      u[s] = scale * z_u[s];
    } else {
      // No pooling: diffuse prior per state (approx "flat") using big_sd_u
      u[s] = big_sd_u * z_u[s];
    }
  }

  // ----- County RE c_j with state-specific scales -----
  vector[J] c;
  for (j in 1:J) {
    real sc = is_treated[j] == 1 ? sigma_c_treat : sigma_c_ctrl;
    c[j] = sc * z_c[j];
  }
  // ----- Per-county treatment effect delta_j -----
  vector[J] delta = rep_vector(0.0, J);
  {
    int k = 1;
    for (j in 1:J) {
      if (is_treated[j] == 1) {
        delta[j] = mu_delta + sigma_delta * z_delta[k];
        k += 1;
      }
    }
  }
}

model {
  // Priors
  A0          ~ normal(0, 5);
  A1          ~ normal(0, 2.5);
  lambda      ~ normal(0, 5);

  sigma_c_treat ~ normal(0, 2.5);
  sigma_c_ctrl  ~ normal(0, 1.5);

  mu_delta    ~ normal(0, 5);
  sigma_delta ~ normal(0, 2.5);

  sigma_y     ~ normal(0, 5);

  // u_s priors:
  if (pool_u_within_group == 1) {
    sigma_u_treat ~ normal(0, 2);
    sigma_u_ctrl  ~ normal(0, 1);
    z_u           ~ normal(0, 1);            // non-centered within-group pooling
  } else {
    // No pooling: very weak N(0, big_sd_u^2) prior per u_s via z_u ~ N(0,1)
    // (Equivalent to u_s ~ N(0, big_sd_u^2))
    z_u ~ normal(0, 1);
  }

  // Non-centered bases
  z_c     ~ normal(0, 1);
  z_delta ~ normal(0, 1);

  // Likelihood
  for (n in 1:N) {
  int j = county[n];
  int s = state_of_county[j];

  real c_use = (center_c == 1) ? (c[j] - mean(c)) : c[j];

  real mu = A0
          + dot_product(row(Xbar, s), A1)
          + u[s]
          + c_use
          + lambda * post[n]
          + delta[j] * D[n];
  
  
  if (is_inf(mu)||is_nan(mu))
  reject("Non-finite mu at obs ", n,
         "; s=", s, ", j=", j,
         "; components: u[s]=", u[s],
         ", c_use=", c_use,
         ", delta[j]=", delta[j],
         ", post=", post[n],
         ", D=", D[n]);

  y[n] ~ normal(mu, sigma_y);
}
}

generated quantities {
  real ATT = mu_delta; // average treatment effect among treated counties
}
